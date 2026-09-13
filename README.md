# antiwindup-tt

Simulation and figure-generation code for the paper

> S. Gulgonul, "Optimal exit from saturation: a closed-form tracking time constant for back-calculation anti-windup under load disturbances", 2026.

The single script `antiwindup_tt_sim.py` reproduces every figure and table of the paper.

## What it contains

| Function | Purpose | Used in |
|---|---|---|
| `sim` | Full discrete-time simulation of PI + back-calculation on a FOPDT process (Algorithm 1 of Caparroz et al., 2026), pulse load disturbance | Figs. 1, 3, 4; Table 2 |
| `alpha_star_sim` | IAE-optimal T_t/T_i by scanning alpha in the full simulation | Figs. 3, 4; Table 2 |
| `dphase` | Reduced normalised recovery-phase model (delay-integrator loop) | Fig. 2 |
| `median_exact` | Integrator impulse response h_i, its median, and the critical level y0* (Theorem 1) | Figs. 2, 3, 4; Table 1 |
| `y0star_nodelay`, `y0star_explicit` | Delay-free closed form (Theorem 3) and explicit approximation, eq. (21) | Fig. 3; Table 1 |
| `alpha_long`, `alpha_short` | The tuning rule for long and short disturbances (Theorem 2 and Section 5.2) | Figs. 4a, 4b; Table 2 |

## Running

```
pip install -r requirements.txt
python antiwindup_tt_sim.py
```

Produces `fig1_time_response.pdf`, `fig2_median.pdf`, `fig3_rslim.pdf`, `fig4_alpha.pdf` and prints the simulated critical saturation ratios. Runtime is a few minutes; the full-simulation scans dominate.

## Parameters

Default process T = 3, L = 0.5 (L/T = 1/6), K = 1, sampling time 0.01 s, unit pulse disturbance. All quantities in the paper are dimensionless (x = lambda/T, L/T, R_S, D_d/T), so other processes are covered by the same code through these ratios.
