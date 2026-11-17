# Adaptation to an AI tutor: practical non-identifiability from success curves

Code accompanying the manuscript *"The Rate of Adult Learners' Adaptation to an AI Tutor Is Not Recoverable From Success Curves: Practical Non-Identifiability and a Withdrawal-Probe Design"*.

All results are produced by a simulation model. No parameter is calibrated to data.

## Files

| File | Purpose |
|---|---|
| `adaptation_model.py` | Base model: adaptation `A(t)` and competence `S(t)`, LSODA integration, cohort sampler |
| `adaptation_model_v2.py` | Extended model: autonomous vs. assisted competence, cognitive offloading, critical offloading `omega*`, optimal adaptation `A*` |
| `fastsim.py` | Vectorised integrator (closed-form `A(t)`, integrating factor for `S(t)`) used by the identifiability analyses; alternative transfer functions |
| `reproduce.py` | Curve matching (Tables 4, 5), adaptation cost grid, cohort statistics, threshold and schedule results (Table 9) |
| `analysis.py` | Integrator check, Fisher information (Table 6), expected profile likelihood, robustness to the transfer function (Table 8) |
| `mc2.py` | Monte Carlo parameter recovery (Table 7) |
| `figures.py` | Figures 1-7 (SVG and PNG) |

## Running

Python 3.11 with NumPy, SciPy and Matplotlib.

```
python reproduce.py      # writes res/reproduce.json, res/lost_grid.npz, res/cohort.npz
python analysis.py       # writes res/analysis.json, res/profiles.npz
python mc2.py            # writes res/mc2.json, res/mc.npz   (about 15 minutes)
python figures.py        # writes fig/*.svg, fig/*.png
```

The scripts expect sibling folders `res/` and `fig/` next to the folder that holds the code. Random seeds are fixed, so the numbers in the manuscript are reproduced exactly.

## Numerical settings

* Reference trajectories: LSODA, rtol 1e-9, atol 1e-11.
* Fast integrator: step 0.01 sessions; maximum deviation from LSODA below 1e-6 in accuracy.
* Fits: bounded scalar minimisation (one parameter) or Nelder-Mead from 9-10 starting points (several parameters), gains on a log scale and offloading on a logit scale.
