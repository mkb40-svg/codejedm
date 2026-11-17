"""Figures for the JEDM manuscript (Times New Roman, colour, SVG + PNG)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import fastsim as fs

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "res")
FIG = os.path.join(HERE, "..", "fig")
rep = json.load(open(os.path.join(RES, "reproduce.json")))
ana = json.load(open(os.path.join(RES, "analysis.json")))

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "savefig.dpi": 300,
    "svg.fonttype": "none",
})
W = 6.3          # text width in inches (letter paper, 2.75 cm side margins)
BLUES = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]   # slow -> fast
ORANGE, GREY, RED = "#d95f02", "#666666", "#b2182b"
TAUS = [r["tau_A"] for r in rep["ident_table"]]
t = fs.TGRID

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)

def tag(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")

# ---------------------------------------------------------------- Figure 1: base dynamics
ref = fs.simulate(v2=False)
Ainf = 0.25 / 0.27
ideal = fs.simulate(v2=False, A0=Ainf)          # already adapted from session 0
fig, ax = plt.subplots(1, 3, figsize=(W, 2.15))
ax[0].plot(t, ref["A"], color=BLUES[1], label=r"adaptation $A(t)$")
ax[0].plot(t, 0.35 + 0.65 * ref["A"], color=ORANGE, ls="--", label=r"transfer $\theta(A)$")
ax[0].set(xlabel="Session", ylabel="Value", ylim=(0, 1.02)); ax[0].legend(frameon=False, loc="lower right")
ax[1].plot(t, ref["S"], color=BLUES[1], label="learner")
ax[1].plot(t, ideal["S"], color=GREY, ls="--", label="already adapted")
ax[1].fill_between(t, ref["S"], ideal["S"], color=BLUES[3], alpha=0.25)
ax[1].set(xlabel="Session", ylabel=r"Competence $S(t)$", ylim=(0, 1.0)); ax[1].legend(frameon=False, loc="lower right")
for a, c, lab in [(0.06, BLUES[0], r"$\tau_A$ = 12.5"), (0.25, BLUES[2], r"$\tau_A$ = 3.7"), (1.2, BLUES[4], r"$\tau_A$ = 0.8")]:
    ax[2].plot(t, fs.simulate(alpha=a, v2=False)["p"], color=c, label=lab)
ax[2].set(xlabel="Session", ylabel=r"Accuracy $p(t)$", ylim=(0.2, 1.0)); ax[2].legend(frameon=False, loc="lower right")
for a_, s_ in zip(ax, "abc"): tag(a_, s_)
fig.tight_layout(w_pad=1.2); save(fig, "fig1_dynamics")

# ---------------------------------------------------------------- Figure 2: degeneracy in the base model
g = np.load(os.path.join(RES, "lost_grid.npz"))
fig, ax = plt.subplots(1, 3, figsize=(W, 2.25))
im = ax[0].pcolormesh(g["et"], g["al"], g["lost"], shading="auto", cmap="viridis")
cb = fig.colorbar(im, ax=ax[0], fraction=0.05, pad=0.03); cb.set_label("Sessions lost", fontsize=9); cb.ax.tick_params(labelsize=8)
ax[0].set(xlabel=r"Learning gain $\eta$", ylabel=r"Adaptation gain $\alpha$")
p_ref = ref["p"]
for r, c in zip(rep["ident_table"], BLUES):
    s = fs.simulate(alpha=r["alpha"], eta=r["eta_fit"], v2=False)
    ax[1].plot(t, s["p"], color=c, lw=1.4)
    ax[2].plot(t, s["A"], color=c, lw=1.4, label=fr"{r['tau_A']:.1f}")
ax[1].set(xlabel="Session", ylabel=r"Accuracy $p(t)$", ylim=(0.2, 1.0)); ax[1].set_title("Accuracy curves coincide", fontsize=9)
ax[2].set(xlabel="Session", ylabel=r"Adaptation $A(t)$", ylim=(0, 1.02)); ax[2].set_title("Adaptation does not", fontsize=9)
ax[2].legend(title=r"$\tau_A$ (sessions)", frameon=False, fontsize=7.5, title_fontsize=8, loc="lower right")
for a_, s_ in zip(ax, "abc"): tag(a_, s_)
fig.tight_layout(w_pad=1.0); save(fig, "fig2_degeneracy")

# ---------------------------------------------------------------- Figure 3: extended model
ref2 = fs.simulate(v2=True)
fig, ax = plt.subplots(1, 3, figsize=(W, 2.25), sharey=True)
for r, c in zip(rep["v2_table"], BLUES):
    s1 = fs.simulate(alpha=r["alpha"], eta=r["eta1"])
    s2 = fs.simulate(alpha=r["alpha"], eta=r["eta2"], omega=r["omega2"])
    ax[0].plot(t, s1["p_assisted"], color=c, lw=1.4)
    ax[1].plot(t, s2["p_assisted"], color=c, lw=1.4)
    ax[2].plot(t, s2["p_unassisted"], color=c, lw=1.4, label=fr"{r['tau_A']:.1f}")
ax[0].set_title(r"$\eta$ free: assisted accuracy", fontsize=9)
ax[1].set_title(r"$\eta,\omega$ free: assisted accuracy", fontsize=9)
ax[2].set_title(r"$\eta,\omega$ free: tutor withdrawn", fontsize=9)
for a_ in ax: a_.set(xlabel="Session", ylim=(0.2, 1.0))
ax[0].set_ylabel("Accuracy")
ax[2].legend(title=r"$\tau_A$ (sessions)", frameon=False, fontsize=7.5, title_fontsize=8, loc="lower right")
for a_, s_ in zip(ax, "abc"): tag(a_, s_)
fig.tight_layout(w_pad=0.8); save(fig, "fig3_extended")

# ---------------------------------------------------------------- Figure 4: practical identifiability
pr = np.load(os.path.join(RES, "profiles.npz"))
tau_grid = 1 / (pr["alpha"] + 0.02)
fig, ax = plt.subplots(1, 2, figsize=(W, 2.7))
styles = {"v1": (GREY, "-", "Base model, accuracy"), "D1": (BLUES[1], "-", "D1: assisted only"),
          "D2": (ORANGE, "--", "D2: + probe at session 30"), "D4": (RED, "-", "D4: + probe every 5 sessions")}
for k, (c, ls, lab) in styles.items():
    ax[0].plot(tau_grid, pr[k], color=c, ls=ls, label=lab)
ax[0].axhline(3.84, color="k", lw=0.8, ls=":"); ax[0].text(0.36, 3.3, "95% level", fontsize=7.5, ha="left", va="top")
ax[0].axvline(1 / 0.27, color="k", lw=0.6, alpha=0.4)
ax[0].set(xscale="log", xlabel=r"$\tau_A$ (sessions)",
          ylabel="Profile deviance", xlim=(0.33, 21), ylim=(0, 13))
ax[0].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
fr = [r for r in ana["fisher"]]
designs = ["v1", "D1", "D2", "D3", "D4", "D5", "D6"]
xs = np.arange(len(designs)); wbar = 0.26
for j, (a_true, c) in enumerate([(0.06, BLUES[0]), (0.25, BLUES[2]), (1.2, BLUES[4])]):
    vals = [next(r["items_needed_factor15"] for r in fr if r["design"] == d and abs(r["alpha"] - a_true) < 1e-9) for d in designs]
    ax[1].bar(xs + (j - 1) * wbar, vals, width=wbar, color=c, label=fr"$\tau_A$ = {1/(a_true+0.02):.1f}")
ax[1].axhline(20, color="k", lw=0.8, ls=":", label="20 items")
ax[1].set(yscale="log", xticks=xs, ylabel="Items needed")
ax[1].set_xticklabels(["Base", "D1", "D2", "D3", "D4", "D5", "D6"])
ax[1].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, columnspacing=1.0, handlelength=1.5)
tag(ax[0], "a"); tag(ax[1], "b")
fig.tight_layout(w_pad=1.5); save(fig, "fig4_identifiability")

# ---------------------------------------------------------------- Figure 5: Monte Carlo recovery
mc = np.load(os.path.join(RES, "mc.npz"))
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
for k, a_true in enumerate((0.25, 0.06)):
    data = [1 / (np.exp(mc[f"a{a_true}_{d}"][:, 0]) + 0.02) for d in ("D1", "D2", "D4")]
    bp = ax[k].boxplot(data, widths=0.5, whis=(10, 90), showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], [BLUES[1], ORANGE, RED]):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for med in bp["medians"]: med.set_color("k")
    rng = np.random.default_rng(1)
    for i, d in enumerate(data):
        ax[k].scatter(i + 1 + rng.uniform(-0.18, 0.18, len(d)), d, s=3, color="k", alpha=0.25, lw=0)
    tt = 1 / (a_true + 0.02)
    ax[k].axhline(tt, color=RED, lw=0.9, ls="--")
    ax[k].set_xticks([1, 2, 3]); ax[k].set_xticklabels(["D1", "D2", "D4"])
    ax[k].set_title(fr"True $\tau_A$ = {tt:.1f} sessions", fontsize=9)
    ax[k].set_yscale("log")
ax[0].set_ylabel(r"Estimated $\tau_A$ (sessions)")
tag(ax[0], "a"); tag(ax[1], "b")
fig.tight_layout(w_pad=1.0); save(fig, "fig5_montecarlo")

# ---------------------------------------------------------------- Figure 6: cohort heterogeneity
c = np.load(os.path.join(RES, "cohort.npz"))
P = c["P"]; sess = np.arange(P.shape[1])
fig, ax = plt.subplots(1, 3, figsize=(W, 2.2))
ax[0].fill_between(sess, np.percentile(P, 10, 0), np.percentile(P, 90, 0), color=BLUES[3], alpha=0.35, lw=0, label="10th-90th pct.")
ax[0].plot(sess, np.median(P, 0), color=BLUES[0], label="median")
ax[0].set(xlabel="Session", ylabel="Accuracy", ylim=(0.2, 1.0)); ax[0].legend(frameon=False, loc="lower right")
ax[1].hist(c["taus"], bins=40, color=BLUES[2], alpha=0.85)
ax[1].axvline(np.median(c["taus"]), color="k", lw=0.9, ls="--")
ax[1].set(xlabel=r"$\tau_A$ (sessions)", ylabel="Learners")
ks = sorted(int(k) for k in rep["pcorr_alpha"])
ax[2].plot(ks, [rep["pcorr_alpha"][str(k)] for k in ks], "o-", color=BLUES[1], ms=3.5, label=r"with $\alpha$ | $\eta$")
ax[2].plot(ks, [rep["pcorr_eta"][str(k)] for k in ks], "s--", color=ORANGE, ms=3.5, label=r"with $\eta$ | $\alpha$")
ax[2].set(xlabel="Session", ylabel="Partial rank correlation", ylim=(0, 1.02)); ax[2].legend(frameon=False, loc="lower right")
for a_, s_ in zip(ax, "abc"): tag(a_, s_)
fig.tight_layout(w_pad=1.0); save(fig, "fig6_cohort")

# ---------------------------------------------------------------- Figure 7: sign of the effect and schedules
from scipy.integrate import solve_ivp
def schedule_traj(w, cut):
    P = dict(fs.BASE); P["omega"] = w
    def rhs(tt, y):
        A, S = y; on = tt < cut
        dA = (P["alpha"] * (1 - A) if on else 0.0) - P["lam_A"] * A
        gain = (P["th0"] + (1 - P["th0"]) * A) * (1 - w * A) if on else P["th0"]
        return [dA, P["eta"] * gain * (1 - S) - P["lam_S"] * S]
    sol = solve_ivp(rhs, (0, 30), [P["A0"], P["S0"]], t_eval=np.linspace(0, 30, 301), rtol=1e-8, atol=1e-10, max_step=0.05)
    return sol.t, sol.y[1]
Agrid = np.linspace(0, 1, 201)
fig, ax = plt.subplots(1, 3, figsize=(W, 2.25))
for w, col in [(0.5, BLUES[1]), (0.683, GREY), (0.8, RED)]:
    ax[0].plot(Agrid, (0.35 + 0.65 * Agrid) * (1 - w * Agrid), color=col, label=fr"$\omega$ = {w}")
ax[0].axhline(0.35, color="k", lw=0.8, ls=":", label=r"control $\theta_0$")
ax[0].set(xlabel=r"Adaptation $A$", ylabel=r"Gain $\theta(A)\varphi(A)$", ylim=(0.15, 0.56)); ax[0].legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.6)
ws = np.linspace(0.3, 0.95, 27)
cont, best = [], []
for w in ws:
    vals = [schedule_traj(w, cut)[1][-1] for cut in range(0, 31)]
    cont.append(vals[30]); best.append(max(vals))
solo = schedule_traj(0.5, 0)[1][-1]
ax[1].plot(ws, cont, color=RED, label="continuous use")
ax[1].plot(ws, best, color=ORANGE, ls="--", label="best cut-off")
ax[1].axhline(solo, color=GREY, lw=1.0, label="no tutor")
ax[1].axvline(ana["robustness"][0]["omega_crit"], color="k", lw=0.6, ls=":")
ax[1].set(xlabel=r"Offloading $\omega$", ylabel=r"$S$ at session 30"); ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")
for cut, col, lab in [(30, RED, "continuous"), (5, ORANGE, "5 sessions, then off"), (0, GREY, "no tutor")]:
    tt, S = schedule_traj(0.8, cut)
    ax[2].plot(tt, S, color=col, ls="--" if cut == 5 else "-", label=lab)
ax[2].set(xlabel="Session", ylabel=r"Autonomous competence $S$", ylim=(0, 0.85)); ax[2].legend(frameon=False, fontsize=7.5, loc="lower right")
ax[2].set_title(r"$\omega$ = 0.8", fontsize=9)
for a_, s_ in zip(ax, "abc"): tag(a_, s_)
fig.tight_layout(w_pad=1.0); save(fig, "fig7_threshold")
print("figures written")
