"""
Does quasi-rigidity predict what a network can detect? (Section 5.2 of the report)

Definition 2 says that a distortion of playback speed is visible on a window of length L when the
distorted motion is at distance more than eps from every natural motion, d_L > eps. This script
tests that prediction on learners that never compute d_L.

For each condition (a view of the walker, a frame rate) and each window length L:
  1. a network is trained from pixels to regress log c on clips of walkers played c times too fast,
     with c known (log c uniform on [log 0.4, log 2.5], plus 25% of natural clips);
  2. on test clips at fixed c, the network detects the distortion when its reading leaves the range
     it takes on natural clips (5% false alarms), which gives a detection rate R_net(c, L);
  3. on the same clips, an explicit observer computes d_L: the root-mean-square distance between the
     keypoints seen and the closest window of a bank of natural-speed clips (no physics involved;
     positions are compared up to a translation when the start position is random). The predicted
     detection rate is R_pred(c, L) = P(d_L > eps), that is 1 - rho in Definition 2.
As a control, the same prediction is made from the size of the distortion alone, |c - 1| V L
(Proposition 1), with V the largest natural speed of the keypoints.

The thresholds (eps for d_L, kappa for the control) are fitted once, on the walker seen by its legs,
by plot_rigidity.py. The other conditions are not used for fitting:
  body only      elastic: neither d_L nor the network should detect anything, whatever eps;
  legs, 20 fps   d_L is a time average, so the boundary in seconds should not move.

Usage
    python rigidity_boundary.py --quick     small check (minutes, CPU)
    python rigidity_boundary.py             run of the report (GPU)
Output
    rigidity.json (or --out), then:  python plot_rigidity.py rigidity.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

H, FINE = 32, 100                    # image size, rate (Hz) at which the bank clips are stored
NOISE = 0.03                         # pixel noise of the videos
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_lin = torch.linspace(-1.2, 1.2, H, device=DEV)
GY, GX = torch.meshgrid(_lin, _lin, indexing="ij")
tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)

# ----------------------------------------------------------------------------------------------
# The walker. Same invented rule as in train_clock.py: the cadence is set by the stride. When the
# body is shown, it translates at a speed also set by the stride.
# ----------------------------------------------------------------------------------------------
f_walk = lambda A: 0.5 + 2*A                 # stride -> cadence (Hz)
v_walk = lambda A: 0.3*A*f_walk(A)           # stride -> speed of the body
# name: (view, positions compared up to translation, spread of the walkers' own pace, frame rate)
CONDITIONS = {
    "legs": ("legs", False, 0.05, 25),
    "body": ("body", True, 0.05, 25),
    "legs, 20 fps": ("legs", False, 0.05, 20),
}

def sample_params(n, rng, jitter):
    """Stride A, own deviation of pace (log), and start position of the body."""
    A = rng.uniform(0.2, 0.8, n); return dict(A=A, dev=rng.normal(0, jitter, n), u0=rng.uniform(0, 1, n))

def geometry(view, p, s):
    """At world times s (n, T): points to draw (n, T, P, 2) and keypoints seen by the observer (n, T, Q, 2).
    Keypoints are the feet for the legs, the hip for the body."""
    A = tt(p["A"])[:, None]
    th = A*torch.sin(2*np.pi*tt(f_walk(p["A"]))[:, None]*s)        # angle of the legs
    if view == "legs":
        hx = torch.zeros_like(s)                                      # hip fixed in the image
    else:                                                             # the body translates; its start is drawn
        disp = tt(v_walk(p["A"]))[:, None]*(s - s[:, :1])             # so that the path stays in view
        span = disp[:, -1:]
        hx = disp - span/2 + (tt(p["u0"])[:, None] - 0.5)*torch.clamp(1.6 - span, min=0)
    hip = torch.stack([hx, torch.full_like(hx, 0.5)], -1)[:, :, None]
    if view == "body":
        return hip, hip
    u = tt(np.linspace(0, 1, 6))
    legs = [hip + torch.stack([torch.sin(sg*th)[..., None]*u, -torch.cos(sg*th)[..., None]*u], -1) for sg in (1, -1)]
    pts = torch.cat(legs, 2)
    feet = torch.stack([legs[0][:, :, -1], legs[1][:, :, -1]], 2)
    return pts, (feet if view == "legs" else torch.cat([hip, feet], 2))

def world_times(p, rate, t, rng):
    """World time of each frame: s = rate * exp(own deviation of pace) * t + random start."""
    n = len(p["dev"])
    return tt(rate*np.exp(p["dev"]))[:, None]*tt(t)[None] + tt(rng.uniform(0, 5, (n, 1)))

def splat(points, sigma):
    """Render points (n, T, P, 2) as Gaussian blobs: images (n, T, H, H)."""
    img = torch.zeros(points.shape[:2] + (H, H), device=DEV)
    for j in range(points.shape[2]):
        px, py = points[:, :, j, 0, None, None], points[:, :, j, 1, None, None]
        img = torch.maximum(img, torch.exp(-((GX - px)**2 + (GY - py)**2)/(2*sigma**2)))
    return img

def clips(cond, log_c, K, rng):
    """Videos (n, 1, K, H, H) of walkers played c times too fast, and the keypoints the observer sees."""
    view, _, jit, fps = CONDITIONS[cond]
    p = sample_params(len(log_c), rng, jit)
    pts, kp = geometry(view, p, world_times(p, np.exp(log_c), np.arange(K)/fps, rng))
    v = splat(pts, 0.1 if view == "body" else 0.07)
    return (v + NOISE*torch.randn_like(v))[:, None], kp

# ----------------------------------------------------------------------------------------------
# Explicit observer: d_L, distance to the closest natural-speed clip of a bank
# ----------------------------------------------------------------------------------------------
def make_bank(cond, n, rng, length=3.5):
    """Keypoints of n natural-speed clips of `length` seconds, stored at FINE Hz: (n, Tf, Q, 2)."""
    view, _, jit, _ = CONDITIONS[cond]
    p = sample_params(n, rng, jit)
    return geometry(view, p, world_times(p, 1.0, np.arange(0, length, 1/FINE), rng))[1]

@torch.no_grad()
def gap(kp, bank, center, fps, chunk=8):
    """d_L of each window kp (m, K, Q, 2): root-mean-square distance, over frames and keypoints, to the
    closest window of the bank, over all bank clips and all start times. Since it is an average over
    frames, it estimates the time integral of Definition 2 and does not depend on the frame rate."""
    K = kp.shape[1]
    idx = torch.arange(K, device=DEV)*(FINE//fps)                             # frame times in the bank's grid
    offs = torch.arange(0, bank.shape[1] - idx[-1], 2, device=DEV)            # candidate start times
    win = bank[:, offs[:, None] + idx[None]]                                  # (N, O, K, Q, 2)
    if center: win = win - win.mean((2, 3), keepdim=True)
    out = []
    for i in range(0, len(kp), chunk):
        w = kp[i:i+chunk]
        if center: w = w - w.mean((1, 2), keepdim=True)
        d = ((win[None] - w[:, None, None])**2).sum(-1).mean((-1, -2))       # (m, N, O)
        out.append(d.flatten(1).min(1).values.sqrt())
    return torch.cat(out)

def size_of_distortion(kp, c, K, fps):
    """Control: |c - 1| V L, the bound of Proposition 1, with V the largest natural speed of the
    keypoints (the apparent speed divided by c)."""
    v_app = (kp[:, 1:] - kp[:, :-1]).norm(dim=-1).amax((1, 2))*fps
    return (abs(c - 1)*v_app/c*K/fps)

# ----------------------------------------------------------------------------------------------
# Network: same architecture as in train_clock.py, with a single output (log c)
# ----------------------------------------------------------------------------------------------
class ClockReader(nn.Module):
    def __init__(self, w=24):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1, w, 3, padding=1), nn.GELU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(w, 2*w, 3, padding=1), nn.GELU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(2*w, 2*w, 3, padding=(2, 1, 1), dilation=(2, 1, 1)), nn.GELU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(2*w, 4*w, 3, padding=(4, 1, 1), dilation=(4, 1, 1)), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(4*w, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return self.f(x)[:, 0]

TRAIN_C = lambda n, rng: np.where(rng.random(n) < 0.25, 0.0, rng.uniform(np.log(0.4), np.log(2.5), n))

def train_net(cond, K, steps, n, rng, width):
    """Train a network to regress log c on windows of K frames."""
    model = ClockReader(width).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for it in range(steps):
        lc = TRAIN_C(n, rng)
        x, _ = clips(cond, lc, K, rng)
        loss = ((model(x) - tt(lc))**2).mean()
        for g in opt.param_groups:                                    # warm-up, then cosine decay
            g["lr"] = 1e-3*min(1.0, (it + 1)/100)*(0.02 + 0.98*0.5*(1 + np.cos(np.pi*it/steps)))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval(), loss.item()

# ----------------------------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def run_condition(cond, Ls, cs, a, rng, log):
    """Detection rate of the network, and d_L and |c - 1| V L of every test clip, on the grid (L, c)."""
    _, center, _, fps = CONDITIONS[cond]
    bank = make_bank(cond, a.bank, rng)
    Ks = [max(3, int(round(L*fps))) for L in Ls]           # same durations in seconds at every frame rate
    res = dict(L=[K/fps for K in Ks], R_net=[], dL=[], size=[])
    for K in Ks:
        t0 = time.time()
        with torch.enable_grad():
            model, last = train_net(cond, K, a.steps, a.batch, rng, a.width)
        # detection threshold: range of the network's readings on natural clips
        nat = []
        for _ in range(4):
            x, _ = clips(cond, np.zeros(a.n_test), K, rng); nat.append(model(x))
        nat = torch.cat(nat); med = nat.median()
        tau = (nat - med).abs().quantile(1 - a.fa)
        Rn, D, S = [], [], []
        for c in cs:
            x, kp = clips(cond, np.full(a.n_test, np.log(c)), K, rng)
            Rn.append(((model(x) - med).abs() > tau).float().mean().item())
            D.append(gap(kp, bank, center, fps).cpu().numpy().tolist())
            S.append(size_of_distortion(kp, c, K, fps).cpu().numpy().tolist())
        res["R_net"].append(Rn); res["dL"].append(D); res["size"].append(S)
        log(f"  {cond:14s} K={K:2d}  loss {last:.3f}  detection " + " ".join(f"{r:.2f}" for r in Rn)
            + f"  ({time.time()-t0:.0f}s)")
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grid and short training, to check the code")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rigidity.json")
    ap.add_argument("--steps", type=int, default=1500, help="training steps per network")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=256, help="test clips per cell of the grid")
    ap.add_argument("--bank", type=int, default=4000, help="natural-speed clips in the observer's bank")
    ap.add_argument("--width", type=int, default=24, help="width of the network")
    ap.add_argument("--cmax", type=float, default=1.35, help="distortions tested: c in [1/cmax, cmax]")
    ap.add_argument("--nc", type=int, default=15, help="number of distortions tested (odd, to include c = 1)")
    ap.add_argument("--fa", type=float, default=0.05, help="false-alarm rate on natural clips")
    a, _ = ap.parse_known_args()
    Ls = [K/25 for K in [4, 6, 8, 12, 16, 24, 32]]          # window lengths (s)
    cs = np.exp(np.linspace(-np.log(a.cmax), np.log(a.cmax), a.nc)).tolist()
    if a.quick:
        Ls, cs = [0.16, 0.32, 0.64], np.exp(np.linspace(-np.log(a.cmax), np.log(a.cmax), 5)).tolist()
        a.steps, a.batch, a.n_test, a.bank = 150, 32, 64, 100
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    torch.backends.cudnn.benchmark = True
    log = lambda s: print(s, flush=True)
    log(f"device {DEV}, windows {Ls} s, steps {a.steps}, batch {a.batch}, bank {a.bank}")
    out = dict(Ls=Ls, cs=cs, fit_on="legs", args=vars(a), conditions={})
    for cond in CONDITIONS:
        out["conditions"][cond] = run_condition(cond, Ls, cs, a, rng, log)
        json.dump(out, open(a.out, "w"))                     # saved after each condition
    log(f"done: {a.out}. Figure and table:  python plot_rigidity.py {a.out}")
