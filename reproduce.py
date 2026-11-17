"""Reproduce the headline numbers of the SIST/Russian manuscript with the
author's own model modules (adaptation_model.py, adaptation_model_v2.py)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr
import adaptation_model as m1
import adaptation_model_v2 as m2

out = {}
T = np.linspace(0, 30, 601)

# ---------------------------------------------------------------- base model
p = m1.Params()
out["tau_A"] = p.tau_A; out["A_inf"] = p.A_inf; out["S_inf"] = p.S_inf
sim = m1.simulate(p, t_end=30, n=601)
out["integrator_err"] = float(np.max(np.abs(sim["A"] - p.A_exact(sim["t"]))))
d, tr, ti = m1.adaptation_cost(p, frac=0.8)
out["dt80"] = d

# grid of lost sessions
al = np.linspace(0.05, 1.0, 40); et = np.linspace(0.05, 0.5, 40)
lost = np.zeros((40, 40)); share = np.zeros((40, 40))
for i, a in enumerate(al):
    for j, e in enumerate(et):
        q = replace(p, alpha=a, eta=e)
        dd, r, idl = m1.adaptation_cost(q, frac=0.8)
        lost[i, j] = dd; share[i, j] = dd / idl
out["lost_range"] = [float(np.nanmin(lost)), float(np.nanmax(lost))]
out["lost_share_max"] = float(np.nanmax(share))
np.savez(os.path.join(os.path.dirname(__file__), "..", "res", "lost_grid.npz"), al=al, et=et, lost=lost, share=share)

# identifiability table (fit eta to reference accuracy for 5 alphas)
ref = m1.simulate(p, t_end=30, n=601)["p_correct"]
alphas = [0.06, 0.10, 0.25, 0.60, 1.20]
rows = []
for a in alphas:
    def sse(e):
        return float(np.sum((m1.simulate(replace(p, alpha=a, eta=e), 30, 601)["p_correct"] - ref) ** 2))
    r = minimize_scalar(sse, bounds=(0.01, 2.0), method="bounded", options={"xatol": 1e-10})
    q = replace(p, alpha=a, eta=r.x)
    s = m1.simulate(q, 30, 601)
    rows.append({"alpha": a, "tau_A": q.tau_A, "eta_fit": r.x,
                 "max_dev_pp": 100 * float(np.max(np.abs(s["p_correct"] - ref))),
                 "A30": float(s["A"][-1])})
out["ident_table"] = rows
A30 = [r_["A30"] for r_ in rows]
Aall = np.array([m1.simulate(replace(p, alpha=r_["alpha"], eta=r_["eta_fit"]), 30, 601)["A"] for r_ in rows])
out["A_spread_max_pp"] = 100 * float(np.max(Aall.max(0) - Aall.min(0)))

# ---------------------------------------------------------------- v2
p2 = m2.P2()
ref2 = m2.simulate(p2)
out["omega_crit"] = p2.omega_crit
out["A_star_w08"] = replace(p2, omega=0.8).A_star

def fit_eta(a):
    def sse(e):
        return float(np.sum((m2.simulate(replace(p2, alpha=a, eta=e))["p_assisted"] - ref2["p_assisted"]) ** 2))
    r = minimize_scalar(sse, bounds=(0.01, 3.0), method="bounded", options={"xatol": 1e-10})
    return r.x

def fit_eta_omega(a):
    best = None
    for le0 in np.log([0.1, 0.18, 0.4]):
        for w0 in [0.2, 0.5, 0.8]:
            x0 = [le0, np.log(w0 / (1 - w0))]
            def sse(x):
                e = np.exp(x[0]); w = 1 / (1 + np.exp(-x[1]))
                return float(np.sum((m2.simulate(replace(p2, alpha=a, eta=e, omega=w))["p_assisted"] - ref2["p_assisted"]) ** 2))
            r = minimize(sse, x0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    return np.exp(best.x[0]), 1 / (1 + np.exp(-best.x[1]))

rows2 = []
for a in alphas:
    e1 = fit_eta(a)
    s1 = m2.simulate(replace(p2, alpha=a, eta=e1))
    e2, w2 = fit_eta_omega(a)
    s2 = m2.simulate(replace(p2, alpha=a, eta=e2, omega=w2))
    rows2.append({"alpha": a, "tau_A": 1 / (a + p2.lam_A),
                  "eta1": e1, "dev1_pp": 100 * float(np.max(np.abs(s1["p_assisted"] - ref2["p_assisted"]))),
                  "eta2": e2, "omega2": w2,
                  "dev2_pp": 100 * float(np.max(np.abs(s2["p_assisted"] - ref2["p_assisted"]))),
                  "rmse2_pp": 100 * float(np.sqrt(np.mean((s2["p_assisted"] - ref2["p_assisted"]) ** 2))),
                  "unassisted30": float(s2["p_unassisted"][-1]),
                  "S30": float(s2["S"][-1])})
out["v2_table"] = rows2
u30 = [r_["unassisted30"] for r_ in rows2]
out["withdrawal_spread_pp"] = 100 * (max(u30) - min(u30))
out["ref_S30"] = float(ref2["S"][-1]); out["ref_unassisted30"] = float(ref2["p_unassisted"][-1])

# threshold checks
for w in [0.683, 0.8]:
    q = replace(p2, omega=w)
    out[f"S_inf_tutor_w{w}"] = q.S_inf(); out[f"S_inf_solo_w{w}"] = q.S_inf(solo=True)
    st = m2.simulate(q); ss = m2.simulate(q, solo=True)
    out[f"S30_tutor_w{w}"] = float(st["S"][-1]); out[f"S30_solo_w{w}"] = float(ss["S"][-1])
    out[f"gap_assisted_unassisted30_w{w}_pp"] = 100 * float(st["p_assisted"][-1] - st["p_unassisted"][-1])

# ---------------------------------------------------------------- cohort
cohort = m1.sample_cohort()
taus = np.array([c.tau_A for c in cohort])
out["cohort_tau"] = {"median": float(np.median(taus)), "q25": float(np.percentile(taus, 25)), "q75": float(np.percentile(taus, 75)),
                     "p05": float(np.percentile(taus, 5)), "p95": float(np.percentile(taus, 95))}
sess = np.arange(0, 31)
P = np.array([m1.simulate(c, 30, 31)["p_correct"] for c in cohort])
out["acc10_p10_p90_pp"] = 100 * float(np.percentile(P[:, 10], 90) - np.percentile(P[:, 10], 10))
la = np.log([c.alpha for c in cohort]); le = np.log([c.eta for c in cohort])
def partial_spearman(x, y, z):
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])
out["pcorr_alpha"] = {int(s): partial_spearman(P[:, s], la, le) for s in [3, 5, 10, 15, 20, 25, 30]}
out["pcorr_eta"] = {int(s): partial_spearman(P[:, s], le, la) for s in [3, 5, 10, 15, 20, 25, 30]}
np.savez(os.path.join(os.path.dirname(__file__), "..", "res", "cohort.npz"), P=P, taus=taus, la=la, le=le)

# ---------------------------------------------------------------- schedules
def run_schedule(q, cut, t_end=30):
    """Tutor used for the first `cut` sessions, then withdrawn: A decays, gain = th0."""
    def rhs(t, y):
        A, S = y
        on = t < cut
        dA = (q.alpha * q.u * (1 - A) if on else 0.0) - q.lam_A * A
        g = float(q.gain(A)) if on else q.th0
        dS = q.eta * q.u * g * (1 - S) - q.lam_S * S
        return [dA, dS]
    ev = [0.0, float(cut), float(t_end)] if 0 < cut < t_end else [0.0, float(t_end)]
    y = [q.A0, q.S0]
    for a_, b_ in zip(ev[:-1], ev[1:]):
        sol = solve_ivp(rhs, (a_, b_), y, rtol=1e-9, atol=1e-11, method="LSODA")
        y = sol.y[:, -1]
    return float(y[1])

sched = []
for w in [0.5, 0.7, 0.8, 0.9]:
    q = replace(p2, omega=w)
    vals = [run_schedule(q, c) for c in range(0, 31)]
    best_c = int(np.argmax(vals)); cont = vals[30]
    lower = run_schedule(replace(q, omega=w - 0.2), 30)
    solo = float(m2.simulate(q, solo=True)["S"][-1])
    sched.append({"omega": w, "best_cut": best_c, "best_S30": vals[best_c], "cont_S30": cont, "solo_S30": solo,
                  "sched_gain_pp": 100 * (vals[best_c] - cont), "lower_gain_pp": 100 * (lower - cont)})
out["schedules"] = sched

json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "res", "reproduce.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
