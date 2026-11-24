"""
Learning to read the clock of a scene from pixels (Section 5.3 of the report).

Every clip of the training corpus has a hidden clock alpha: one second of playback shows alpha
seconds of the world. A small spatio-temporal CNN reads log(alpha), with an uncertainty, on the
video of a single object. The same network is trained in two ways on the same uncalibrated videos:

  baseline      the self-supervised principle of playback-speed predictors (SpeedNet, Pulse of
                Motion): each clip is resampled by a known factor rho and the network predicts
                log(rho). This implicitly assumes that the source clips play at natural speed.

  shared clock  the network is anchored on one object with exact labels (walker clips at natural
                speed, resampled). Then, in unlabelled clips where two objects share the same hidden
                clock, the clock read on the known object is the regression target for the new one:
                walker -> looper (and walker -> snail, an elastic control), then looper -> pulser.

Test: clips of each object alone, under hidden clocks log(alpha) ~ N(0.2, 0.45^2). Reported per
object: bias and standard deviation of the error on log(alpha), and the uncertainty the network
predicts for itself.

Usage
    python train_clock.py                       small run (a few minutes, CPU is enough)
    python train_clock.py --big --seed 0        run of the report (GPU)
Output
    results.json (or --out): errors and statistics per method and per object.
"""
import argparse, copy, json, time
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------
# The world. Each object follows a rule linking its shape to its pace. The rules are invented, so
# that nothing can be inferred from physics, and they are never given to the network.
# ----------------------------------------------------------------------------------------------
FPS, T, H = 25, 32, 32               # playback frame rate, frames per clip, image size (pixels)
MU_A, SD_A = 0.2, 0.45               # hidden clocks of the corpus: log alpha ~ N(MU_A, SD_A^2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_lin = torch.linspace(-1.2, 1.2, H, device=DEV)
GY, GX = torch.meshgrid(_lin, _lin, indexing="ij")            # pixel coordinates

def splat(points, sigma=0.08):
    """Render points of shape (n, T, P, 2) as Gaussian blobs: images of shape (n, T, H, H)."""
    img = torch.zeros(points.shape[:2] + (H, H), device=DEV)
    for j in range(points.shape[2]):
        px, py = points[:, :, j, 0, None, None], points[:, :, j, 1, None, None]
        img = torch.maximum(img, torch.exp(-((GX - px)**2 + (GY - py)**2)/(2*sigma**2)))
    return img

def sample_params(kind, n, rng):
    """Shape of each object, and its own small deviation from the rule ('dev', in log pace)."""
    if kind == "walker":                 # stride A; cadence f(A) = 0.5 + 2A
        A = rng.uniform(0.2, 0.8, n); return dict(A=A, w=0.5 + 2*A, dev=rng.normal(0, 0.05, n))
    if kind == "looper":                 # radius r; angular speed 3/r
        r = rng.uniform(0.4, 1.0, n); return dict(r=r, w=3.0/r, dev=rng.normal(0, 0.05, n))
    if kind == "pulser":                 # size R; pulsing frequency 0.6 + 1.5R
        R = rng.uniform(0.3, 0.9, n); return dict(R=R, w=0.6 + 1.5*R, dev=rng.normal(0, 0.07, n))
    if kind == "snail":                  # straight line; its speed is linked to nothing (elastic)
        return dict(v=np.exp(rng.normal(np.log(0.25), 0.6, n)), ang=rng.uniform(0, 2*np.pi, n), dev=np.zeros(n))
    raise ValueError(kind)

def render(kind, p, log_clock, rng):
    """Video of each object seen through the clock: world time s = alpha * exp(dev) * t + start."""
    n = len(log_clock)
    tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)
    rate = tt(np.exp(log_clock + p["dev"]))                      # world seconds per playback second
    s = rate[:, None]*tt(np.arange(T)/FPS)[None] + tt(rng.uniform(0, 5, (n, 1)))
    if kind == "walker":                                         # two legs swinging around a fixed hip
        th = tt(p["A"])[:, None]*torch.sin(2*np.pi*tt(p["w"])[:, None]*s)
        u = tt(np.linspace(0, 1, 6))
        legs = []
        for sg in (1, -1):
            dx, dy = torch.sin(sg*th)[..., None]*u, -torch.cos(sg*th)[..., None]*u
            legs.append(torch.stack([dx, 0.5 + dy], -1))
        return splat(torch.cat(legs, 2), 0.07)
    if kind == "looper":                                         # a dot on a circle of radius r
        ang = tt(p["w"])[:, None]*s; r = tt(p["r"])[:, None]
        return splat(torch.stack([r*torch.cos(ang), r*torch.sin(ang)], -1)[:, :, None], 0.1)
    if kind == "pulser":                                         # a disc whose radius oscillates
        rad = tt(p["R"])[:, None]*(1 + 0.3*torch.sin(2*np.pi*tt(p["w"])[:, None]*s))
        dist = torch.sqrt(GX**2 + GY**2)
        return torch.sigmoid((rad[:, :, None, None] - dist[None, None])/0.04)
    if kind == "snail":                                          # a dot moving straight
        x = tt(p["v"])[:, None]*(s - s[:, :1]) - 0.6; ang = tt(p["ang"])[:, None]
        return splat(torch.stack([x*torch.cos(ang), x*torch.sin(ang)], -1)[:, :, None], 0.1)

def clips(kind, log_clock, rng, params=None):
    """Noisy videos of shape (n, 1, T, H, H), one object each, under the given clocks."""
    p = sample_params(kind, len(log_clock), rng) if params is None else params
    v = render(kind, p, log_clock, rng)
    v = v + 0.03*torch.randn_like(v)
    return v[:, None], p

# ----------------------------------------------------------------------------------------------
# The model: reads log(alpha) on the video of one object, with a log-variance.
# Temporal dilations let it see the whole clip; pooling is spatial only.
# ----------------------------------------------------------------------------------------------
class ClockReader(nn.Module):
    def __init__(self, w=24):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1, w, (3, 3, 3), padding=1), nn.GELU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(w, 2*w, (3, 3, 3), padding=1), nn.GELU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(2*w, 2*w, (3, 3, 3), padding=(2, 1, 1), dilation=(2, 1, 1)), nn.GELU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(2*w, 4*w, (3, 3, 3), padding=(4, 1, 1), dilation=(4, 1, 1)), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(4*w, 64), nn.GELU(), nn.Linear(64, 2))
    def forward(self, x):
        out = self.f(x); return out[:, 0], out[:, 1].clamp(-5, 2)     # mean, log-variance

def nll(mean, logvar, target):
    """Squared error for the mean, plus a Gaussian likelihood for the variance computed with the
    mean detached, so that the variance cannot slow down the learning of the mean."""
    return ((target - mean)**2 + 0.5*(logvar + (target - mean.detach())**2/logvar.exp())).mean()

# ----------------------------------------------------------------------------------------------
# Training batches
# ----------------------------------------------------------------------------------------------
RESAMPLE = lambda n, rng: rng.uniform(np.log(0.4), np.log(3.5), n)   # known resampling factors (log)

def batch_baseline(rng, n, kinds):
    """Baseline: uncalibrated clips, resampled by a known factor rho; the target is log(rho) only."""
    k = rng.choice(kinds)
    hidden = rng.normal(MU_A, SD_A, n)                 # clock of the source clip, unknown to the learner
    if k == "walker" and rng.random() < 0.5:           # the baseline also gets the walker's natural-speed clips
        hidden = np.zeros(n)
    rho = RESAMPLE(n, rng)
    x, _ = clips(k, hidden + rho, rng)
    return x, torch.tensor(rho, dtype=torch.float32, device=DEV)

def batch_anchor(rng, n):
    """Shared clock, anchor: walker clips at natural speed, resampled, hence with exact labels."""
    rho = RESAMPLE(n, rng)
    x, _ = clips("walker", rho, rng)
    return x, torch.tensor(rho, dtype=torch.float32, device=DEV)

def batch_pair(rng, n, known, new, model, dev):
    """Shared clock, transfer: two objects under the same hidden clock; the clock read on the known
    object (by the frozen model) is the target for the new one. No label is used."""
    la = rng.normal(MU_A, SD_A, n)
    xk, _ = clips(known, la, rng); xn, _ = clips(new, la, rng)
    with torch.no_grad():
        target, _ = model(xk)
    return xn, target.clamp(-3, 3)

def train(regime, steps, n, dev, rng, log, width=24):
    model = ClockReader(width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    phases = (["baseline"] if regime == "baseline" else ["anchor", "walker>looper+snail", "looper>pulser"])
    for phase, st in zip(phases, steps):
        t0 = time.time()
        # Targets are read by a copy of the model frozen at the start of the phase, so that they do
        # not move while the new object is being learned.
        teacher = copy.deepcopy(model).eval()
        for it in range(st):
            if phase == "baseline":
                x, y = batch_baseline(rng, n, ["walker", "looper", "pulser", "snail"])
            elif phase == "anchor":
                x, y = batch_anchor(rng, n)
            else:
                r = rng.random()
                if r < 0.35:                                              # the anchor is kept throughout
                    x, y = batch_anchor(rng, n)
                elif phase == "walker>looper+snail":
                    x, y = batch_pair(rng, n, "walker", "looper" if r < 0.75 else "snail", teacher, dev)
                else:                                                     # new pair, plus replay of the earlier ones
                    if r < 0.75:   x, y = batch_pair(rng, n, "looper", "pulser", teacher, dev)
                    elif r < 0.90: x, y = batch_pair(rng, n, "walker", "looper", teacher, dev)
                    else:          x, y = batch_pair(rng, n, "walker", "snail", teacher, dev)
            mean, logvar = model(x.to(dev))
            loss = nll(mean, logvar, y.to(dev))
            for g in opt.param_groups:                                    # warm-up, then cosine decay within each phase
                g["lr"] = 1e-3*min(1.0, (it + 1)/100)*(0.02 + 0.98*0.5*(1 + np.cos(np.pi*it/st)))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if it % max(1, st//5) == 0:
                log(f"  [{regime}/{phase}] step {it}/{st}  loss {loss.item():.3f}  ({time.time()-t0:.0f}s)")
        if regime == "baseline": break
        r = evaluate(model, dev, rng, 256)                               # bias / spread after each phase
        log("  after " + phase + ": " + "  ".join(f"{k} {v['bias']:+.2f}/{v['std']:.2f}" for k, v in r.items()))
    return model

@torch.no_grad()
def evaluate(model, dev, rng, n_test):
    """Read the clock on clips of each object alone, under hidden clocks drawn from the corpus."""
    model.eval(); res = {}
    for k in ["walker", "looper", "pulser", "snail"]:
        la = rng.normal(MU_A, SD_A, n_test)
        errs, sds = [], []
        for i in range(0, n_test, 64):
            x, p = clips(k, la[i:i+64], rng)
            mean, logvar = model(x.to(dev))
            errs.append(mean.cpu().numpy() - la[i:i+64]); sds.append(np.exp(0.5*logvar.cpu().numpy()))
        e, s = np.concatenate(errs), np.concatenate(sds)
        res[k] = dict(bias=float(e.mean()), std=float(e.std()), rmse=float(np.sqrt((e**2).mean())),
                      predicted_std=float(s.mean()), errors=e.tolist())
    model.train(); return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="settings of the report (batch 128, 3 x 3000 steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--width", type=int, default=24, help="width of the network")
    ap.add_argument("--steps", type=int, default=None, help="steps per phase (overrides the preset)")
    a, _ = ap.parse_known_args()
    dev = DEV
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if a.big: n, steps, n_test = 128, [3000, 3000, 3000], 2000
    else:     n, steps, n_test = 32, [400, 400, 400], 400
    if a.steps: steps = [a.steps]*3
    log = lambda s: print(s, flush=True)
    log(f"device {dev}, batch {n}, steps {steps}")
    out = {}
    log("training baseline (resampling, assumes natural speed)")
    out["baseline"] = evaluate(train("baseline", [sum(steps)], n, dev, rng, log, a.width), dev, rng, n_test)
    log("training shared clock (anchor on walker, then walker -> looper/snail -> pulser)")
    out["shared"] = evaluate(train("shared", steps, n, dev, rng, log, a.width), dev, rng, n_test)
    for reg in out:
        for k, r in out[reg].items():
            log(f"{reg:9s} {k:7s} bias {r['bias']:+.3f}  std {r['std']:.3f}  predicted std {r['predicted_std']:.3f}")
    json.dump(out, open(a.out, "w"))
