"""Monte Carlo parameter recovery with a more robust maximum-likelihood search.

Each simulated learner answers K = 20 binomial items per measurement occasion.
The likelihood is maximised over (log alpha, log eta, logit omega) from ten
starting points; a fit is flagged as an optimiser failure if its negative
log-likelihood is worse than the value at the true parameters.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
import fastsim as fs

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res")
os.makedirs(RES, exist_ok=True)
K, R = 20, 200
LOG15 = np.log(1.5)

def make_design(name):
    assisted = [("pa", k) for k in range(1, 31)]
    return {"D1": assisted,
            "D2": assisted + [("pu", 30)],
            "D4": assisted + [("pu", k) for k in range(5, 31, 5)]}[name]

def predict(x, design):
    s = fs.simulate(alpha=np.exp(x[0]), eta=np.exp(x[1]), omega=1 / (1 + np.exp(-x[2])))
    vals = [s["p_assisted" if k == "pa" else "p_unassisted"][int(round(t / fs.DT))] for k, t in design]
    return np.clip(np.array(vals), 1e-9, 1 - 1e-9)

def nll(x, des, y):
    if not (np.log(0.003) < x[0] < np.log(30)) or not (np.log(0.003) < x[1] < np.log(30)) or abs(x[2]) > 25:
        return 1e12
    q = predict(x, des)
    return -float(np.sum(y * np.log(q) + (K - y) * np.log(1 - q)))

STARTS = [[np.log(a), np.log(0.18), w] for a in (0.03, 0.1, 0.3, 1.0, 3.0) for w in (-1.0, 1.0)]

def mle(des, y):
    best = None
    for x0 in STARTS:
        r = minimize(nll, x0, args=(des, y), method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    r = minimize(nll, best.x, args=(des, y), method="Nelder-Mead",
                 options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 3000})
    return (r.x, r.fun) if r.fun < best.fun else (best.x, best.fun)

rng = np.random.default_rng(20261117)
out, raw = [], {}
t0 = time.time()
for a_true in (0.25, 0.06):
    x_true = np.array([np.log(a_true), np.log(0.18), 0.0])
    for name in ("D1", "D2", "D4"):
        des = make_design(name)
        p_true = predict(x_true, des)
        est, fails = [], 0
        for rep in range(R):
            y = rng.binomial(K, p_true).astype(float)
            x, f = mle(des, y)
            if f > nll(x_true, des, y) + 1e-6:
                fails += 1
            est.append(x)
        est = np.array(est)
        err = np.abs(est[:, 0] - np.log(a_true))
        tau_hat = 1 / (np.exp(est[:, 0]) + 0.02)
        row = {"alpha": a_true, "tau_A": 1 / (a_true + 0.02), "design": name, "optimizer_failures": fails,
               "within_factor15": float(np.mean(err <= LOG15)), "within_factor2": float(np.mean(err <= np.log(2))),
               "median_abs_log_err": float(np.median(err)),
               "tau_hat_q10_q50_q90": [float(np.percentile(tau_hat, q)) for q in (10, 50, 90)]}
        out.append(row); raw[f"a{a_true}_{name}"] = est
        print(row, f"({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(os.path.join(RES, "mc2.json"), "w"), indent=1)
np.savez(os.path.join(RES, "mc.npz"), **raw)
print("done")
