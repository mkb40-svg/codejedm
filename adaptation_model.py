"""
Two-component model of the rate of cognitive adaptation to an AI tutor
(adult learners).  Continuous time t is measured in *sessions*.

State
-----
A(t) in [0,1)  adaptation to the tutor: the fraction of the tutor's
               instructional potential the learner can actually exploit
               (prompt/query quality, calibrated trust, help-seeking,
               interpreting generated explanations).
S(t) in [0,1)  subject-matter competence (latent mastery).

Dynamics (practice intensity u(t) >= 0, sessions per unit time)

    dA/dt = alpha * u * (1 - A)                 - lam_A * A
    dS/dt = eta   * u * theta(A) * (1 - S)      - lam_S * S
    theta(A) = th0 + (1 - th0) * A

theta(A) is the *instructional transfer efficiency*: at A = 0 only a
fraction th0 of what the tutor delivers becomes learning; at A = 1 all of
it does.  Adaptation therefore acts as a time-varying gain on the learning
channel, not as an additive bonus.

The primary quantity of the study -- the rate of cognitive adaptation --
is the eigenvalue of the A-equation under constant intensity u:

    r_A   = alpha * u + lam_A        [1 / session]
    tau_A = 1 / r_A                  [sessions]
    A_inf = alpha * u / (alpha * u + lam_A)

with the exact solution  A(t) = A_inf + (A0 - A_inf) * exp(-r_A * t).
The S-equation is linear in S given A(t) but has no elementary closed
form, so it is integrated numerically; the analytic A is used to verify
the integrator.

Observation models (what an ITS log or a test actually records)
    accuracy        p(t) = g + (1 - g) * S(t)
    item latency    T(t) = T_min + (T_max - T_min) * (1 - S(t))
    interaction cost C(t) = C_min + (C_max - C_min) * (1 - A(t))
                            (turns needed to resolve one impasse)

Author's note: all parameter values here are illustrative ranges, not
empirical estimates.  Nothing in this module is calibrated to data.
"""

from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    alpha: float = 0.25   # adaptation gain per session
    lam_A: float = 0.02   # decay of adaptation (tutor updates, disuse)
    th0:   float = 0.35   # transfer efficiency of an unadapted learner
    eta:   float = 0.18   # intrinsic learning gain per session
    lam_S: float = 0.01   # forgetting of competence
    A0:    float = 0.15   # prior familiarity with AI tutors
    S0:    float = 0.10   # prior domain knowledge
    g:     float = 0.20   # guessing floor of the assessment
    u:     float = 1.00   # practice intensity (sessions per unit time)

    # ---- analytic quantities of the A-subsystem -------------------------
    @property
    def r_A(self) -> float:
        """Rate of cognitive adaptation [1/session]."""
        return self.alpha * self.u + self.lam_A

    @property
    def tau_A(self) -> float:
        """Adaptation time constant [sessions]."""
        return 1.0 / self.r_A

    @property
    def A_inf(self) -> float:
        return self.alpha * self.u / self.r_A

    @property
    def theta_inf(self) -> float:
        return self.th0 + (1.0 - self.th0) * self.A_inf

    @property
    def S_inf(self) -> float:
        """Competence asymptote once adaptation has equilibrated."""
        k = self.eta * self.u * self.theta_inf
        return k / (k + self.lam_S)

    def A_exact(self, t):
        t = np.asarray(t, float)
        return self.A_inf + (self.A0 - self.A_inf) * np.exp(-self.r_A * t)


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------

def rhs(t, y, p: Params, frozen_A=None):
    A, S = y
    A_eff = A if frozen_A is None else frozen_A
    theta = p.th0 + (1.0 - p.th0) * A_eff
    dA = p.alpha * p.u * (1.0 - A) - p.lam_A * A
    dS = p.eta * p.u * theta * (1.0 - S) - p.lam_S * S
    return [dA, dS]


def simulate(p: Params, t_end=40.0, n=801, frozen_A=None):
    """Integrate the model.  frozen_A=x holds the transfer gain at theta(x),
    i.e. a counterfactual learner who is already adapted (use A_inf)."""
    t = np.linspace(0.0, t_end, n)
    y0 = [p.A0, p.S0]
    sol = solve_ivp(rhs, (0.0, t_end), y0, args=(p, frozen_A),
                    t_eval=t, dense_output=True, rtol=1e-9, atol=1e-11,
                    method="LSODA")
    A, S = sol.y
    return {"t": t, "A": A, "S": S,
            "p_correct": p.g + (1.0 - p.g) * S,
            "sol": sol}


def time_to_fraction(p: Params, frac=0.5, t_end=400.0, frozen_A=None):
    """First time S(t) reaches frac * S_inf.  Returns np.nan if never."""
    target = frac * p.S_inf
    if p.S0 >= target:
        return 0.0
    sol = solve_ivp(rhs, (0.0, t_end), [p.A0, p.S0], args=(p, frozen_A),
                    dense_output=True, rtol=1e-9, atol=1e-11, method="LSODA")
    f = lambda tt: sol.sol(tt)[1] - target
    if f(t_end) < 0:
        return np.nan
    return brentq(f, 0.0, t_end, xtol=1e-8)


def adaptation_cost(p: Params, frac=0.5, t_end=400.0):
    """Sessions lost to the adaptation transient: the delay in reaching
    frac*S_inf relative to a learner who starts already adapted (A == A_inf).
    Both systems share the same asymptote S_inf, so the comparison is a pure
    delay."""
    t_real = time_to_fraction(p, frac, t_end)
    t_ideal = time_to_fraction(p, frac, t_end, frozen_A=p.A_inf)
    return t_real - t_ideal, t_real, t_ideal


# --------------------------------------------------------------------------
# cohort heterogeneity
# --------------------------------------------------------------------------

def sample_cohort(n=2000, base: Params = Params(), seed=20260101,
                  sd_log_alpha=0.50, sd_log_eta=0.45, rho=0.40):
    """Log-normal alpha and eta with a shared aptitude correlation rho;
    Beta-distributed prior knowledge and prior AI familiarity."""
    rng = np.random.default_rng(seed)
    cov = np.array([[sd_log_alpha**2, rho * sd_log_alpha * sd_log_eta],
                    [rho * sd_log_alpha * sd_log_eta, sd_log_eta**2]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    alpha = base.alpha * np.exp(z[:, 0])
    eta = base.eta * np.exp(z[:, 1])
    A0 = rng.beta(2.0, 10.0, n)
    S0 = rng.beta(2.0, 12.0, n)
    return [replace(base, alpha=a, eta=e, A0=a0, S0=s0)
            for a, e, a0, s0 in zip(alpha, eta, A0, S0)]


if __name__ == "__main__":
    p = Params()
    print(asdict(p))
    print(f"r_A={p.r_A:.4f}/session  tau_A={p.tau_A:.2f} sessions  "
          f"A_inf={p.A_inf:.3f}  S_inf={p.S_inf:.3f}")
