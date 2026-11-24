# What does it mean for a model to understand time? Code

This folder contains the code for the report *A foundation model that truly understands time* (van der Schaar Lab research task). The videos are synthetic and are generated on the fly, so no data needs to be downloaded.

| file | what it does | in the report |
|---|---|---|
| `time_concepts.ipynb` | Numerical checks of the definitions and results with an explicit observer: a bank of natural-speed clips, no physics, no training. | Section 5.1 |
| `rigidity_boundary.py` | Trains networks from pixels to detect playback distortions and compares where they succeed with the prediction of quasi-rigidity, $d_L > \varepsilon$. | Section 5.2 |
| `plot_rigidity.py` | Figure 2 and the gaps between predicted and observed detection rates. | Figure 2 |
| `plot_frames.py` | Frames of the four objects of `train_clock.py`. | Figure 1 |
| `train_clock.py` | Trains a network to read the clock of a scene, with the resampling baseline or with the shared clock, and tests it on each object alone. | Section 5.3 |
| `plot_clock.py` | Figure 3 and Table 1, from several seeds. | Figure 3, Table 1 |

## Installation

```
pip install -r requirements.txt
```

A GPU is recommended for the full runs. Each script also has a small setting that runs on a CPU in a few minutes.

## Reproducing the figures

Section 5.1 runs the notebook from top to bottom. Every result is printed or plotted below the cell that computes it.

Section 5.2 takes well under an hour on one GPU:
```
python rigidity_boundary.py                  # writes rigidity.json
python plot_rigidity.py rigidity.json        # writes rigidity.png, rigidity_all.png, rigidity_table.md
```
The threshold $\varepsilon$ is fitted once, on the walker seen by its legs, and then kept fixed for the other conditions (body only, and 20 frames per second). `--bank 400` reproduces the smaller bank mentioned in the text.

Section 5.3 uses 10 seeds, each taking a few minutes on one GPU:
```
for s in 0 1 2 3 4 5 6 7 8 9; do python train_clock.py --big --seed $s --out results_$s.json; done
python plot_clock.py results_*.json          # writes seeds.png, seeds_table.md, seeds_table.tex
```
Use `python train_clock.py` without `--big` for a short run.

Results vary slightly from run to run on a GPU, because some GPU operations are not deterministic even with a fixed seed.

Figure 1: `python plot_frames.py` (writes `frames.png`).

The `results/` folder contains the raw outputs behind the report: `s0.json` to `s9.json` for Figure 3 and Table 1, `rigidity_main.json` (bank of 4000 clips) for Figure 2, and `rigidity_small_bank.json` (bank of 400 clips) for the gap of 0.116 mentioned in Section 5.2. Running the plotting scripts on these files reproduces the figures without any training. Both rigidity files also contain a condition not discussed in the report (a walker with a more irregular pace).

## The synthetic world

Each object links its shape to its pace by a rule that the learner is never given:
- the walker's cadence is set by its stride;
- the looper's angular speed is set by its radius;
- the pulser's frequency is set by its size;
- the snail's speed is set by nothing, which makes it an elastic control.

The rules are invented, so that nothing can be inferred from physics. Every clip has a hidden clock $\alpha$, with $\log\alpha \sim \mathcal N(0.2, 0.45^2)$: one second of playback shows $\alpha$ seconds of the world.