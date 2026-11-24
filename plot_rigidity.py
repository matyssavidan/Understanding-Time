"""
Figure 2 and the numbers of Section 5.2, from the output of rigidity_boundary.py.

The threshold eps of d_L (and kappa of the control |c - 1| V L) is fitted once, on the walker seen
by its legs, by minimising the mean absolute gap between predicted and observed detection rates over
the grid (cells with c = 1 excluded). It is then kept fixed for every other condition.

Usage
    python plot_rigidity.py rigidity.json
Writes
    rigidity.png        Figure 2: legs, rate predicted by d_L next to the network's rate
    rigidity_all.png    the same pair for every condition
    rigidity_table.md   gap between predicted and observed rates, per condition
"""
import json, sys
import numpy as np
import matplotlib.pyplot as plt

def rates(values, thr):
    """Predicted detection rate per cell of the grid: fraction of test clips whose value exceeds thr.
    values has shape (windows, distortions, clips)."""
    return (np.asarray(values) > thr).mean(-1)

def fit_threshold(values, R_net, mask):
    """Threshold minimising the mean |predicted - observed| detection rate over the cells in mask."""
    v = np.asarray(values)
    grid = np.quantile(v[v > 0], np.linspace(0.001, 0.999, 400)) if (v > 0).any() else [1e-6]
    err = [np.abs(rates(v, g) - R_net)[mask].mean() for g in grid]
    return float(grid[int(np.argmin(err))])

def analyse(out):
    """Fit eps and kappa on the reference condition, then measure the gap on every condition, with the
    fitted thresholds and, for comparison, with thresholds refitted on that condition."""
    cs = np.array(out["cs"])
    mask = np.broadcast_to(np.abs(np.log(cs)) > 1e-9, (len(out["Ls"]), len(cs)))
    fit = out["fit_on"]; R = {k: np.array(v["R_net"]) for k, v in out["conditions"].items()}
    eps = fit_threshold(out["conditions"][fit]["dL"], R[fit], mask)
    kap = fit_threshold(out["conditions"][fit]["size"], R[fit], mask)
    table = {}
    for k, v in out["conditions"].items():
        mad = lambda vals, thr: float(np.abs(rates(vals, thr) - R[k])[mask].mean())
        table[k] = dict(dL=mad(v["dL"], eps), size=mad(v["size"], kap),
                        dL_refit=mad(v["dL"], fit_threshold(v["dL"], R[k], mask)),
                        size_refit=mad(v["size"], fit_threshold(v["size"], R[k], mask)))
    return eps, kap, table

def pair(axes, v, cs, eps, kap, titles):
    """Two maps of detection rate over (window L, distortion c): predicted by d_L, and observed.
    Red line: predicted rate 1/2. Dashed blue (right map): the same from the size of the distortion."""
    L = np.array(v["L"])
    Rp, Rs, Rn = rates(v["dL"], eps), rates(v["size"], kap), np.array(v["R_net"])
    for ax, R, title in zip(axes, [Rp, Rn], titles):
        im = ax.pcolormesh(L, cs, R.T, vmin=0, vmax=1, cmap="Greys", shading="nearest")
        if Rp.min() < 0.5 < Rp.max():
            ax.contour(L, cs, Rp.T, levels=[0.5], colors="C3", linewidths=1.2)
        ax.set_yscale("log"); ax.set_xlabel(r"window $L$ (s)"); ax.set_title(title, fontsize=8)
        ticks = [t for t in [0.8, 0.9, 1.0, 1.1, 1.25] if cs.min() <= t <= cs.max()]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    if Rs.min() < 0.5 < Rs.max():
        axes[1].contour(L, cs, Rs.T, levels=[0.5], colors="C0", linestyles="--", linewidths=1.2)
    axes[0].set_ylabel(r"distortion $c$")
    return im

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "rigidity.json"
    out = json.load(open(path))
    eps, kap, table = analyse(out)
    cs = np.array(out["cs"]); conds = out["conditions"]
    plt.rcParams.update({"font.size": 8})

    # Figure 2: the reference condition
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.1), sharey=True, constrained_layout=True)
    im = pair(axes, conds[out["fit_on"]], cs, eps, kap,
              [r"predicted: $\pi(d_L>\varepsilon)$", "network trained from pixels"])
    fig.colorbar(im, ax=axes, label="detection rate", fraction=0.05, pad=0.02)
    fig.savefig("rigidity.png", dpi=250)

    # every condition, one (predicted, network) pair per row
    fig, axes = plt.subplots(len(conds), 2, figsize=(5.2, 2.1*len(conds)), sharey=True,
                             constrained_layout=True, squeeze=False)
    for row, (k, v) in zip(axes, conds.items()):
        tag = " (fit)" if k == out["fit_on"] else ""
        im = pair(row, v, cs, eps, kap, [f"{k}{tag}: predicted", f"network, gap {table[k]['dL']:.3f}"])
    fig.colorbar(im, ax=axes, label="detection rate", fraction=0.04, pad=0.02)
    fig.savefig("rigidity_all.png", dpi=200)

    # table: mean |predicted - observed| detection rate over the grid (c != 1)
    lines = [f"eps = {eps:.4f}, kappa = {kap:.4f}, fitted on '{out['fit_on']}'.",
             "Mean |predicted - network detection rate| over the grid (c != 1).", "",
             "| condition | d_L, eps fixed | size, kappa fixed | d_L, refitted | size, refitted |",
             "|---|---|---|---|---|"]
    lines += [f"| {k}{' (fit)' if k == out['fit_on'] else ''} | {t['dL']:.3f} | {t['size']:.3f} "
              f"| {t['dL_refit']:.3f} | {t['size_refit']:.3f} |" for k, t in table.items()]
    open("rigidity_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
