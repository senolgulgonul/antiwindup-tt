"""Table 3: IAE penalty of T_t = T_i and of the instantaneous reset relative to
the proposed rule, for the cases of Table 2 (L/T = 1/6).

Uses the full discrete-time simulation `sim` of antiwindup_tt_sim.py
(Algorithm 1 of Caparroz et al., T = 3, L = 0.5, K = 1, Ts = 0.01).
The rule values of alpha are those of Table 2 (rule column)."""
from antiwindup_tt_sim import sim

CASES = [  # x, R_S, {D_d/T: alpha from the rule}
    (0.2, 0.6, {1/3: 0.15, 1: 0.42, 10: 0.59}),
    (0.2, 0.8, {1/3: 0.41, 1: 0.64, 10: 0.77}),
    (0.2, 0.9, {1/3: 0.61, 1: 0.76, 10: 0.84}),
    (1.0, 0.8, {1/3: 0.00, 1: 0.10, 10: 0.28}),
    (1.0, 0.9, {1/3: 0.08, 1: 0.34, 10: 0.50}),
]

if __name__ == "__main__":
    print("x    R_S  | increase of IAE in per cent, (T_t = T_i)/(instantaneous reset)")
    for x, Rs, rule in CASES:
        cells = []
        for DdT, a in rule.items():
            J = sim(x, Rs, DdT, a)[0]        # rule
            J1 = sim(x, Rs, DdT, 1.0)[0]     # T_t = T_i
            J0 = sim(x, Rs, DdT, 0.0)[0]     # instantaneous reset (T_t = T_s)
            cells.append(f"D_d/T={DdT:5.2f}: {100*(J1/J-1):4.1f}/{100*(J0/J-1):4.1f}")
        print(f"{x:3.1f}  {Rs:3.1f}  | " + "   ".join(cells))
