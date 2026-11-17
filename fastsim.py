"""Fast, vectorised simulator of the author's model (v1 and v2).

A(t) has a closed form; S(t) obeys a linear ODE given A(t),
    S' = a(t) (1 - S) - lam_S S,   a(t) = eta*u*theta(A)*phi(A),
which is solved with an integrating factor and cumulative trapezoid
quadrature on a fine grid.  Agreement with the LSODA reference solution
of adaptation_model(_v2).py is checked in analysis.py.
"""
import numpy as np

BASE = dict(alpha=0.25, lam_A=0.02, th0=0.35, eta=0.18, lam_S=0.01,
            omega=0.50, kappa=0.60, A0=0.15, S0=0.10, g=0.20, u=1.0)

DT = 0.01
T_END = 30.0
TGRID = np.arange(0.0, T_END + DT / 2, DT)
SESS_IDX = np.rint(np.arange(1, 31) / DT).astype(int)      # observations at sessions 1..30


def theta_fn(A, th0, form=("lin",)):
    kind = form[0]
    if kind == "lin":
        return th0 + (1 - th0) * A
    if kind == "pow":                       # th0 + (1-th0) A^gamma
        return th0 + (1 - th0) * np.power(np.clip(A, 0, None), form[1])
    if kind == "sat":                       # concave saturating, k = form[1]
        k = form[1]
        return th0 + (1 - th0) * (1 - np.exp(-k * A)) / (1 - np.exp(-k))
    if kind == "sig":                       # sigmoidal around A = 0.5, slope form[1]
        k = form[1]
        s = lambda x: 1 / (1 + np.exp(-k * (x - 0.5)))
        return th0 + (1 - th0) * (s(A) - s(0.0)) / (s(1.0) - s(0.0))
    raise ValueError(kind)


def simulate(alpha=None, eta=None, omega=None, v2=True, form=("lin",), t=TGRID, **kw):
    P = dict(BASE); P.update(kw)
    if alpha is not None: P["alpha"] = alpha
    if eta is not None: P["eta"] = eta
    if omega is not None: P["omega"] = omega
    r = P["alpha"] * P["u"] + P["lam_A"]
    Ainf = P["alpha"] * P["u"] / r
    A = Ainf + (P["A0"] - Ainf) * np.exp(-r * t)
    th = theta_fn(A, P["th0"], form)
    ph = (1 - P["omega"] * A) if v2 else np.ones_like(A)
    a = P["eta"] * P["u"] * th * ph
    b = a + P["lam_S"]
    dt = np.diff(t)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (b[1:] + b[:-1]) * dt)])
    f = a * np.exp(I)
    J = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dt)])
    S = np.exp(-I) * (P["S0"] + J)
    g = P["g"]
    out = {"t": t, "A": A, "S": S, "p_unassisted": g + (1 - g) * S}
    if v2:
        out["p_assisted"] = g + (1 - g) * np.clip(S + P["kappa"] * A * (1 - S), 0, 1)
    else:
        out["p"] = g + (1 - g) * S
    return out
