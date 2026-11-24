"""
Figure 1 of the report: frames of the four objects of train_clock.py, at natural speed and without
noise, every third frame.

Usage
    python plot_frames.py            writes frames.png
"""
import numpy as np
import matplotlib.pyplot as plt
from train_clock import sample_params, render

rng = np.random.default_rng(0)
objs = {"walker": "walker (anchor)", "looper": "looper", "pulser": "pulser", "snail": "snail"}
shown = list(range(0, 24, 3))                                   # frames t_0, t_3, ..., t_21

fig, axes = plt.subplots(len(objs), len(shown), figsize=(1.15*len(shown), 1.2*len(objs)),
                         gridspec_kw=dict(wspace=0.04, hspace=0.06))
for row, (kind, name) in zip(axes, objs.items()):
    p = sample_params(kind, 1, rng)
    video = render(kind, p, np.zeros(1), rng)[0].cpu().numpy()  # clock log(alpha) = 0: natural speed
    for ax, k in zip(row, shown):
        ax.imshow(video[k], cmap="gray", vmin=0, vmax=1, origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    row[0].set_ylabel(name, fontsize=9)
for ax, k in zip(axes[0], shown):
    ax.set_title(f"$t_{{{k}}}$", fontsize=9)
fig.savefig("frames.png", dpi=200, bbox_inches="tight")