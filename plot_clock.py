"""
Figure 3 and Table 1 of the report, from several runs of train_clock.py (one per seed).

Usage
    for s in 0 1 2 3 4 5 6 7 8 9; do python train_clock.py --big --seed $s --out results_$s.json; done
    python plot_clock.py results_*.json
Writes
    seeds.png        Figure 3: bias and RMSE on log(alpha) per object, every seed and mean +- std
    seeds_table.md   mean +- std over seeds of bias, spread, RMSE and predicted uncertainty
    seeds_table.tex  rows of Table 1
"""
import json, re, sys
import numpy as np
import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in sys.argv[1:]]
objs = ["walker", "looper", "pulser", "snail"]
names = {"walker": "walker (anchor)", "looper": "looper (via walker)",
         "pulser": "pulser (via looper)", "snail": "snail (elastic)"}
methods = {"baseline": "resampling baseline", "shared": "shared clock"}

def stat(method, obj, key):
    """One value per seed."""
    return np.array([r[method][obj][key] for r in runs])

# ---------- table, Markdown: mean +- standard deviation across seeds ----------
lines = [f"Results over {len(runs)} seeds (mean ± standard deviation across seeds).", "",
         "| object | method | bias | std of error | RMSE | predicted std |", "|---|---|---|---|---|---|"]
for k in objs:
    for m in methods:
        b, s, e, p = (stat(m, k, key) for key in ["bias", "std", "rmse", "predicted_std"])
        lines.append(f"| {names[k]} | {methods[m]} | {b.mean():+.3f} ± {b.std():.3f} | {s.mean():.3f} ± {s.std():.3f} "
                     f"| {e.mean():.3f} ± {e.std():.3f} | {p.mean():.3f} ± {p.std():.3f} |")
# seeds in which the whole chain works: the shared clock beats the baseline on both learned objects
win = [(r["shared"]["looper"]["rmse"] < r["baseline"]["looper"]["rmse"]) and
       (r["shared"]["pulser"]["rmse"] < r["baseline"]["pulser"]["rmse"]) for r in runs]
lines += ["", f"Shared clock better than the baseline in RMSE on both learned objects: {sum(win)}/{len(runs)} seeds."]
open("seeds_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------- table, LaTeX rows of Table 1 ----------
f = lambda x, sign: re.sub(r"^([+-]?)0\.", r"\1.", f"{x:+.3f}" if sign else f"{x:.3f}")   # -0.160 -> -.160
cell = lambda v, sign: f"${f(v.mean(), sign)}_{{\\pm{f(v.std(), False)}}}$"
tex = [f"{k:6s} & " + " & ".join([cell(stat("baseline", k, "bias"), True), cell(stat("shared", k, "bias"), True),
                                   cell(stat("baseline", k, "rmse"), False), cell(stat("shared", k, "rmse"), False)]) + r" \\"
       for k in objs]
tex.append("% predicted uncertainty, shared: " + ", ".join(f(stat("shared", k, "predicted_std").mean(), False) for k in objs)
           + "; baseline: " + ", ".join(f(stat("baseline", k, "predicted_std").mean(), False) for k in objs))
open("seeds_table.tex", "w").write("\n".join(tex) + "\n")

# ---------- Figure 3 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
x = np.arange(len(objs))
jitter = np.random.default_rng(0).uniform(-0.04, 0.04, (len(runs), len(objs)))
for m, col, off in [("baseline", "C0", -0.12), ("shared", "C1", 0.12)]:
    for ax, key in [(axes[0], "bias"), (axes[1], "rmse")]:
        vals = np.array([[r[m][k][key] for k in objs] for r in runs])              # (seeds, objects)
        ax.scatter((x[None] + off + jitter).ravel(), vals.ravel(), s=12, c=col, alpha=.6,
                   label=methods[m] if key == "bias" else None)                    # one dot per seed
        ax.errorbar(x + off, vals.mean(0), yerr=vals.std(0), fmt="o", c=col, ms=6, capsize=3, mec="k")
axes[0].set_title(r"bias of the error on $\log\alpha$", fontsize=9)
axes[1].set_title(r"RMSE on $\log\alpha$", fontsize=9)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels([names[k] for k in objs], fontsize=8); ax.axhline(0, c="k", lw=.6)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("seeds.png", dpi=200)
