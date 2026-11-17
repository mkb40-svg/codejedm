"""
Model v2: adaptation to an AI tutor with *autonomous* competence separated
from assisted performance.

Revision of v1 (adaptation_model.py).  v1 had a single competence S whose
only fate was to grow, so it could not reproduce the central empirical
finding of the field -- that the measured advantage of a generative tutor
disappears when the tutor is taken away (Bastani et al. 2024,
10.2139/ssrn.4895486).  v2 makes two changes:

1. S(t) now means *autonomous* competence: what the learner can do with the
   tutor switched off.  Performance *with* the tutor is a separate,
   observable quantity that includes competence borrowed from the system.

2. Adaptation A(t) has two opposing consequences.  It raises the
   instructional transfer efficiency theta(A) -- a better-adapted learner
   extracts more from the same tutor -- and it raises cognitive offloading
   phi(A) -- a better-adapted learner hands more of the work over.  Only
   the work the learner still does themselves builds autonomous competence.

State
-----
A(t)  adaptation to the tutor (as in v1)
S(t)  autonomous competence

    dA/dt = alpha*u*(1-A) - lam_A*A
    dS/dt = eta*u * theta(A) * phi(A) * (1-S) - lam_S*S

    theta(A) = th0 + (1-th0)*A      transfer efficiency, increasing
    phi(A)   = 1 - omega*A          germane share, decreasing (omega = offloading

A solo control learner (no tutor) has the same equation with the gain fixed
at th0: working alone is exactly as efficient as working with a tutor you
cannot yet use.  So the tutor can only help through theta, and can only harm
through phi.  The net gain factor is theta(A)*phi(A) against th0 for the
control, which makes the harm condition explicit:

    theta(A_inf)*phi(A_inf) < th0   =>   the tutored learner ends up with
                                         LESS autonomous competence

    omega* = (1 - th0/theta(A_inf)) / A_inf      critical offloading

Observation models
------------------
    with tutor     p_assisted(t)   = g + (1-g) * [S + kappa*A*(1-S)]
    tutor withdrawn p_unassisted(t) = g + (1-g) * S
    borrowed competence D(t)        = kappa*A(t)*(1-S(t))

D(t) is what collapses at withdrawal.  It is proportional to A(t), which is
why periodic withdrawal probes identify the adaptation rate that assisted
accuracy alone cannot (see v1 identifiability result).

All parameter values are illustrative.  Nothing here is calibrated to data.
"""

from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class P2:
    alpha: float = 0.25    # adaptation gain
    lam_A: float = 0.02    # decay of adaptation
    th0:   float = 0.35    # transfer efficiency of an unadapted learner
    eta:   float = 0.18    # intrinsic learning gain
    lam_S: float = 0.01    # forgetting of autonomous competence
    omega: float = 0.50    # offloading: share of germane work handed over at A=1
    kappa: float = 0.60    # how much of the gap the tutor fills during assisted work
    A0:    float = 0.15
    S0:    float = 0.10
    g:     float = 0.20
    u:     float = 1.00

    @property
    def r_A(self):   return self.alpha * self.u + self.lam_A
    @property
    def tau_A(self): return 1.0 / self.r_A
    @property
    def A_inf(self): return self.alpha * self.u / self.r_A

    def theta(self, A): return self.th0 + (1.0 - self.th0) * np.asarray(A, float)
    def phi(self, A):   return 1.0 - self.omega * np.asarray(A, float)
    def gain(self, A):  return self.theta(A) * self.phi(A)

    @property
    def omega_crit(self):
        """Offloading above which the tutored learner ends below the control."""
        return (1.0 - self.th0 / float(self.theta(self.A_inf))) / self.A_inf

    @property
    def A_star(self):
        """Adaptation level that maximises the autonomous-learning gain."""
        if self.omega <= 0:
            return 1.0
        a = (1.0 - self.th0) * self.omega
        x = ((1.0 - self.th0) - self.th0 * self.omega) / (2.0 * a)
        return float(np.clip(x, 0.0, 1.0))

    def S_inf(self, solo=False):
        k = self.eta * self.u * (self.th0 if solo else float(self.gain(self.A_inf)))
        return k / (k + self.lam_S)

    def A_exact(self, t):
        t = np.asarray(t, float)
        return self.A_inf + (self.A0 - self.A_inf) * np.exp(-self.r_A * t)


def rhs(t, y, p: P2, frozen_A=None, solo=False):
    A, S = y
    gain = p.th0 if solo else float(p.gain(A if frozen_A is None else frozen_A))
    dA = p.alpha * p.u * (1.0 - A) - p.lam_A * A
    dS = p.eta * p.u * gain * (1.0 - S) - p.lam_S * S
    return [dA, dS]


def simulate(p: P2, t_end=30.0, n=601, frozen_A=None, solo=False):
    t = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs, (0.0, t_end), [p.A0, p.S0], args=(p, frozen_A, solo),
                    t_eval=t, dense_output=True, rtol=1e-9, atol=1e-11, method="LSODA")
    A, S = sol.y
    borrowed = p.kappa * A * (1.0 - S)
    return {"t": t, "A": A, "S": S, "borrowed": borrowed,
            "p_assisted":   p.g + (1 - p.g) * np.clip(S + borrowed, 0, 1),
            "p_unassisted": p.g + (1 - p.g) * S,
            "sol": sol}


def time_to_fraction(p: P2, frac=0.8, t_end=600.0, frozen_A=None, solo=False):
    target = frac * p.S_inf(solo=solo)
    if p.S0 >= target:
        return 0.0
    sol = solve_ivp(rhs, (0.0, t_end), [p.A0, p.S0], args=(p, frozen_A, solo),
                    dense_output=True, rtol=1e-9, atol=1e-11, method="LSODA")
    f = lambda tt: sol.sol(tt)[1] - target
    if f(t_end) < 0:
        return np.nan
    return brentq(f, 0.0, t_end, xtol=1e-8)


def adaptation_cost(p: P2, frac=0.8, t_end=600.0):
    """Sessions lost (>0) or gained (<0) because adaptation takes time,
    relative to a learner already at A_inf from session 0."""
    t_real = time_to_fraction(p, frac, t_end)
    t_ideal = time_to_fraction(p, frac, t_end, frozen_A=p.A_inf)
    return t_real - t_ideal, t_real, t_ideal
