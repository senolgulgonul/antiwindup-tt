"""IAE penalty of T_t = T_i and of the instantaneous reset relative to the
proposed rule, for the cases of Table 2 (L/T = 1/6).

Normalised continuous-time model of the paper (time in units of T),
forward-Euler integration; the instantaneous reset is applied exactly.
Rule values of alpha are those of Table 2 (rule column)."""
import numpy as np

DT = 2e-4

def iae(x, l, RS, D, alpha, dt=DT):
    g = 1.0 / (x + l); dhat = 1.0 / (1.0 - RS); nl = int(round(l / dt))
    n = int((D + l + 40 * max(1.0, 1.0 / g)) / dt)
    buf = np.zeros(n + nl + 1); y = 0.0; ui = 0.0; J = 0.0
    for k in range(n):
        uc = -g * y + ui
        sat = uc < -1.0
        buf[k + nl] = max(uc, -1.0) + (dhat if k * dt < D else 0.0)
        if sat:
            if alpha <= dt:
                ui = -1.0 + g * y                      # instantaneous reset
            else:
                ui += dt * (-g * y - (1.0 + ui - g * y) / alpha)
        else:
            ui += dt * (-g * y)
        y += dt * (-y + buf[k])
        J += abs(y) * dt
    return J

L = 1.0 / 6.0
cases = [  # x, RS, {D: alpha_rule}
    (0.2, 0.6, {1/3: 0.15, 1: 0.42, 10: 0.59}),
    (0.2, 0.8, {1/3: 0.41, 1: 0.64, 10: 0.77}),
    (0.2, 0.9, {1/3: 0.61, 1: 0.76, 10: 0.84}),
    (1.0, 0.8, {1/3: 0.00, 1: 0.10, 10: 0.28}),
    (1.0, 0.9, {1/3: 0.08, 1: 0.34, 10: 0.50}),
]
if __name__ == "__main__":
    print("x  RS  | D: +%(Tt=Ti) / +%(reset)")
    for x, RS, d in cases:
        out = []
        for D, a in d.items():
            J = iae(x, L, RS, D, a)
            J1 = iae(x, L, RS, D, 1.0)
            J0 = iae(x, L, RS, D, 0.0)
            out.append(f"D={D:.2f}: {100*(J1/J-1):5.1f}/{100*(J0/J-1):5.1f}")
        print(x, RS, " | ", "   ".join(out))
