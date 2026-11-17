"""New analyses for the JEDM version.

1. Check of the fast simulator against the LSODA reference.
2. Local identifiability: Fisher information of (log alpha, log eta[, logit omega])
   under binomial item noise for several measurement designs.
3. Expected profile likelihood of alpha (practical identifiability).
4. Monte Carlo parameter recovery for three designs.
5. Robustness of the degeneracy to the functional form of theta(A).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize, minimize_scalar
import fastsim as fs
import adaptation_model as m1
import adaptation_model_v2 as m2

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res")
os.makedirs(RES, exist_ok=True)
out = {}
K = 20                      # test items per measurement occasion
LOG15 = np.log(1.5)

# ------------------------------------------------------------------ 1. check
s1 = fs.simulate(v2=False)
r1 = m1.simulate(m1.Params(), t_end=30, n=3001)
s2 = fs.simulate(v2=True)
r2 = m2.simulate(m2.P2(), t_end=30, n=3001)
out["fastsim_err_v1"] = float(np.max(np.abs(s1["p"] - r1["p_correct"])))
out["fastsim_err_v2_assisted"] = float(np.max(np.abs(s2["p_assisted"] - r2["p_assisted"])))
out["fastsim_err_v2_unassisted"] = float(np.max(np.abs(s2["p_unassisted"] - r2["p_unassisted"])))

# 44 pp check: slowest fitted v1 model vs reference in A
ref1 = fs.simulate(v2=False)
def fit_eta_v1(a, form=("lin",), ref=None):
    ref = ref1["p"] if ref is None else ref
    f = lambda e: float(np.sum((fs.simulate(alpha=a, eta=e, v2=False, form=form)["p"] - ref) ** 2))
    return minimize_scalar(f, bounds=(0.005, 3.0), method="bounded", options={"xatol": 1e-10}).x
e_slow = fit_eta_v1(0.06)
A_slow = fs.simulate(alpha=0.06, eta=e_slow, v2=False)["A"]
out["A_maxdiff_slowest_vs_ref_pp"] = 100 * float(np.max(np.abs(A_slow - ref1["A"])))
out["A_maxdiff_slowest_vs_ref_t"] = float(fs.TGRID[np.argmax(np.abs(A_slow - ref1["A"]))])
e_fast = fit_eta_v1(1.2)
A_fast = fs.simulate(alpha=1.2, eta=e_fast, v2=False)["A"]
out["A_maxdiff_fastest_vs_ref_pp"] = 100 * float(np.max(np.abs(A_fast - ref1["A"])))

# ------------------------------------------------------------------ helpers
def unpack(x, v2):
    a, e = np.exp(x[0]), np.exp(x[1])
    w = 1 / (1 + np.exp(-x[2])) if v2 else None
    return a, e, w

def predict(x, design, v2=True, form=("lin",)):
    a, e, w = unpack(x, v2)
    s = fs.simulate(alpha=a, eta=e, omega=w, v2=v2, form=form)
    vals = []
    for kind, sess in design:
        idx = int(round(sess / fs.DT))
        key = {"p": "p", "pa": "p_assisted", "pu": "p_unassisted"}[kind]
        vals.append(s[key][idx])
    return np.clip(np.array(vals), 1e-9, 1 - 1e-9)

def make_design(name):
    assisted = [("pa", k) for k in range(1, 31)]
    if name == "v1":  return [("p", k) for k in range(1, 31)]
    if name == "D1":  return assisted
    if name == "D2":  return assisted + [("pu", 30)]
    if name == "D3":  return assisted + [("pu", k) for k in (10, 20, 30)]
    if name == "D4":  return assisted + [("pu", k) for k in range(5, 31, 5)]
    if name == "D5":  return assisted + [("pu", k) for k in (2, 4, 6, 8, 10)]
    if name == "D6":  return [("pu", k) for k in range(5, 31, 5)]
    raise ValueError(name)

DESIGN_LABEL = {
    "v1": "Base model, accuracy at sessions 1-30",
    "D1": "Assisted accuracy at sessions 1-30",
    "D2": "D1 + one withdrawal probe at session 30",
    "D3": "D1 + withdrawal probes at sessions 10, 20, 30",
    "D4": "D1 + withdrawal probes every 5 sessions",
    "D5": "D1 + early probes at sessions 2, 4, 6, 8, 10",
    "D6": "Withdrawal probes every 5 sessions only",
}
out["design_labels"] = DESIGN_LABEL

def fisher(x, design, v2=True, k_items=K, h=1e-5):
    p0 = predict(x, design, v2)
    Jc = []
    for i in range(len(x)):
        xp, xm = np.array(x, float), np.array(x, float)
        xp[i] += h; xm[i] -= h
        Jc.append((predict(xp, design, v2) - predict(xm, design, v2)) / (2 * h))
    J = np.array(Jc).T
    W = k_items / (p0 * (1 - p0))
    return J.T @ (W[:, None] * J), J

def logit(w): return np.log(w / (1 - w))

# ------------------------------------------------------------------ 2. Fisher information
fim_rows = []
for a_true in (0.06, 0.25, 1.20):
    for name in ("v1", "D1", "D2", "D3", "D4", "D5", "D6"):
        v2 = name != "v1"
        x0 = [np.log(a_true), np.log(0.18)] + ([logit(0.5)] if v2 else [])
        F, J = fisher(x0, make_design(name), v2)
        cov = np.linalg.inv(F)
        se = np.sqrt(np.diag(cov))
        corr = cov / np.outer(se, se)
        cosJ = float(np.dot(J[:, 0], J[:, 1]) / np.linalg.norm(J[:, 0]) / np.linalg.norm(J[:, 1]))
        target = LOG15 / 1.96
        k_req = K * (se[0] / target) ** 2
        fim_rows.append({"alpha": a_true, "tau_A": 1 / (a_true + 0.02), "design": name,
                         "se_log_alpha": float(se[0]), "corr_alpha_eta": float(corr[0, 1]),
                         "corr_alpha_omega": float(corr[0, 2]) if v2 else None,
                         "cos_sens_alpha_eta": cosJ,
                         "cond": float(np.linalg.cond(F)),
                         "items_needed_factor15": float(k_req)})
out["fisher"] = fim_rows

# ------------------------------------------------------------------ 3. expected profile likelihood
def exp_deviance(q, p, k_items=K):
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return 2 * k_items * float(np.sum(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))))

alpha_grid = np.exp(np.linspace(np.log(0.03), np.log(3.0), 41))
profiles = {}
x_true_v2 = [np.log(0.25), np.log(0.18), logit(0.5)]
for name in ("v1", "D1", "D2", "D4"):
    v2 = name != "v1"
    des = make_design(name)
    p_true = predict(x_true_v2 if v2 else x_true_v2[:2], des, v2)
    prof = []
    for a in alpha_grid:
        if not v2:
            f = lambda le: exp_deviance(predict([np.log(a), le], des, False), p_true)
            r = minimize_scalar(f, bounds=(np.log(0.005), np.log(3.0)), method="bounded", options={"xatol": 1e-9})
            prof.append(float(r.fun))
        else:
            best = np.inf
            for le0 in (np.log(0.1), np.log(0.18), np.log(0.4)):
                for w0 in (0.2, 0.5, 0.9):
                    f = lambda z: exp_deviance(predict([np.log(a), z[0], z[1]], des, True), p_true)
                    r = minimize(f, [le0, logit(w0)], method="Nelder-Mead",
                                 options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 3000})
                    best = min(best, float(r.fun))
            prof.append(best)
    prof = np.array(prof)
    inside = alpha_grid[prof <= 3.84]
    profiles[name] = prof
    out[f"profile_{name}"] = {"ci95_alpha": [float(inside.min()), float(inside.max())] if len(inside) else None,
                              "ci95_tau": [float(1 / (inside.max() + 0.02)), float(1 / (inside.min() + 0.02))] if len(inside) else None,
                              "dev_at_half_alpha": float(np.interp(np.log(0.125), np.log(alpha_grid), prof)),
                              "dev_at_double_alpha": float(np.interp(np.log(0.5), np.log(alpha_grid), prof))}
np.savez(os.path.join(RES, "profiles.npz"), alpha=alpha_grid, **profiles)

# ------------------------------------------------------------------ 4. Monte Carlo recovery
def negll(x, des, y, n, v2=True):
    q = predict(x, des, v2)
    return -float(np.sum(y * np.log(q) + (n - y) * np.log(1 - q)))

def mle(des, y, n):
    best = None
    for a0 in (0.08, 0.3, 1.0):
        x0 = [np.log(a0), np.log(0.18), 0.0]
        r = minimize(negll, x0, args=(des, y, n), method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 2500})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x.copy()
    x[0] = np.clip(x[0], np.log(0.005), np.log(20.0))
    return x

rng = np.random.default_rng(20261117)
R = 200
mc = []
mc_raw = {}
t0 = time.time()
for a_true in (0.25, 0.06):
    x_true = [np.log(a_true), np.log(0.18), logit(0.5)]
    for name in ("D1", "D2", "D4"):
        des = make_design(name)
        p_true = predict(x_true, des)
        est = []
        for rep in range(R):
            y = rng.binomial(K, p_true)
            est.append(mle(des, y, K))
        est = np.array(est)
        la = est[:, 0]
        err = np.abs(la - np.log(a_true))
        tau_hat = 1 / (np.exp(la) + 0.02)
        mc.append({"alpha": a_true, "tau_A": 1 / (a_true + 0.02), "design": name,
                   "within_factor15": float(np.mean(err <= LOG15)),
                   "within_factor2": float(np.mean(err <= np.log(2))),
                   "median_abs_log_err": float(np.median(err)),
                   "tau_hat_q10_q50_q90": [float(np.percentile(tau_hat, q)) for q in (10, 50, 90)]})
        mc_raw[f"a{a_true}_{name}"] = est
        print(f"MC alpha={a_true} {name}: {mc[-1]}  ({time.time()-t0:.0f}s)", flush=True)
out["monte_carlo"] = mc
np.savez(os.path.join(RES, "mc.npz"), **mc_raw)

# ------------------------------------------------------------------ 5. robustness to theta(A)
forms = {"linear": ("lin",), "power, gamma=0.5": ("pow", 0.5), "power, gamma=2": ("pow", 2.0),
         "saturating, k=4": ("sat", 4.0), "sigmoidal, k=8": ("sig", 8.0)}
rob = []
for label, form in forms.items():
    ref = fs.simulate(v2=False, form=form)["p"]
    devs = []
    for a in (0.06, 0.10, 0.60, 1.20):
        e = fit_eta_v1(a, form, ref)
        devs.append(100 * float(np.max(np.abs(fs.simulate(alpha=a, eta=e, v2=False, form=form)["p"] - ref))))
    refv2 = fs.simulate(v2=True, form=form)
    v2dev, v2wd = [], []
    for a in (0.06, 0.10):
        best = None
        for le0 in (np.log(0.1), np.log(0.18), np.log(0.4)):
            for w0 in (0.2, 0.5, 0.8):
                f = lambda z: float(np.sum((fs.simulate(alpha=a, eta=np.exp(z[0]), omega=1/(1+np.exp(-z[1])), form=form)["p_assisted"] - refv2["p_assisted"]) ** 2))
                r = minimize(f, [le0, logit(w0)], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000})
                if best is None or r.fun < best.fun:
                    best = r
        s = fs.simulate(alpha=a, eta=np.exp(best.x[0]), omega=1 / (1 + np.exp(-best.x[1])), form=form)
        v2dev.append(100 * float(np.max(np.abs(s["p_assisted"] - refv2["p_assisted"]))))
        v2wd.append(100 * float(np.max(np.abs(s["p_unassisted"] - refv2["p_unassisted"]))))
    Ainf = 0.25 / 0.27
    th_inf = float(fs.theta_fn(np.array([Ainf]), 0.35, form)[0])
    rob.append({"form": label, "v1_max_dev_pp": max(devs), "v1_devs_pp": devs,
                "v2_assisted_max_dev_pp_slow": max(v2dev), "v2_withdrawal_sep_pp_slow": max(v2wd),
                "omega_crit": (1 - 0.35 / th_inf) / Ainf})
out["robustness"] = rob

json.dump(out, open(os.path.join(RES, "analysis.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "design_labels"}, indent=1, default=float))
