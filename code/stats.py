"""Statistics for experiment 2: do cheap scores rank planners the way reactive closed-loop safety does?

Input: the tidy per-run table written by extract_metrics.py (results/runs.csv), one row per
planner x protocol x scenario x seed, with columns
    planner, family, protocol, scenario, scenario_type, seed, valid, <metric columns>
Protocols: ols (open loop, with captured trajectories), navsim (NAVSIM scorer at the first frame), clsnr (closed loop,
non-reactive agents), clsr (closed loop, reactive agents), interplan (interPlan benchmark, its own 80 scenarios).

Conventions
- Every score is oriented so that higher = better / safer before ranking (sign column in CHEAP / TARGETS).
- Planner-level values are means over the scenarios of one scenario set. Cheap scores and targets are computed on the
  same 700 Val14 scenarios (same-scenario design); interPlan is a separate scenario set.
- Kendall tau-b throughout (ties allowed). Bootstrap CIs are percentile intervals; scenario resampling is stratified by
  scenario type and paired across cheap score and target.
Output: CSV tables in results/stats/ and a dict that run_stats() returns (merged into results/summary.json).
"""
import json
import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge

# name -> (protocol, column, sign); sign +1 if higher is better, -1 if lower is better
CHEAP = {
    "L2": ("ols", "l2_avg", -1),
    "LogCollision": ("ols", "log_collision_rate", -1),
    "OLS": ("ols", "ols_score", +1),
    "PDMS": ("navsim", "pdms", +1),
    "EPDMS": ("navsim", "epdms", +1),
    "CLS-NR": ("clsnr", "cls_score", +1),
}
# Safety outcomes in nuPlan's reactive closed loop (per scenario: 1 = violation, severity in [0, 1])
TARGETS = {
    "AtFaultCollision": ("clsr", "at_fault_collision", -1),
    "TTCViolation": ("clsr", "ttc_violation", -1),
    "DrivableAreaViolation": ("clsr", "drivable_violation", -1),
    "DirectionViolation": ("clsr", "direction_violation", -1),
    "AnySafetyViolation": ("clsr", "any_violation", -1),
    "SeverityWeighted": ("clsr", "severity_s1", -1),
    "ViolationOrNoProgress": ("clsr", "violation_or_no_progress", -1),
}
REFERENCE = {"CLS-R": ("clsr", "cls_score", +1), "Progress": ("clsr", "m_progress_ratio", +1)}
INTERPLAN = ("interplan", "interplan_score", +1)
# Worst value per column, used when invalid runs are counted as failures (sensitivity analysis)
WORST = {"ols_score": 0.0, "pdms": 0.0, "epdms": 0.0, "cls_score": 0.0, "interplan_score": 0.0,
         "log_collision_rate": 1.0, "at_fault_collision": 1.0, "ttc_violation": 1.0, "drivable_violation": 1.0,
         "direction_violation": 1.0, "any_violation": 1.0, "severity_s1": 1.0, "severity_s2": 1.0,
         "violation_or_no_progress": 1.0, "m_progress_ratio": 0.0}

B_MAIN, B_SLICE, B_LOFO, B_COUNT = 2000, 500, 300, 300
SEED = 20261028


# ----------------------------------------------------------------------------------------------- basic statistics
def tau_b(a, b):
    """Kendall tau-b for 1-D arrays, or row-wise for 2-D arrays (replicates x planners). NaN-free input."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    i, j = np.triu_indices(a.shape[1], 1)
    da, db = np.sign(a[:, i] - a[:, j]), np.sign(b[:, i] - b[:, j])
    n1, n2 = (da != 0).sum(1), (db != 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (da * db).sum(1) / np.sqrt(n1 * n2)
    return t if t.size > 1 else float(t[0])


def inversion_rate(a, b):
    """Share of planner pairs ordered differently by a and b, among pairs strictly ordered by both."""
    i, j = np.triu_indices(len(a), 1)
    da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    ok = (da != 0) & (db != 0)
    return float((da[ok] != db[ok]).mean()) if ok.any() else float("nan")


def mmrv(proxy, target):
    """Mean Maximum Rank Violation (Li et al., SIMPLER): for each planner, the largest target difference to another
    planner that the proxy orders the wrong way; averaged over planners. In the target's own units (lower is better)."""
    i, j = np.meshgrid(np.arange(len(proxy)), np.arange(len(proxy)), indexing="ij")
    wrong = np.sign(proxy[i] - proxy[j]) * np.sign(target[i] - target[j]) < 0
    return float((np.abs(target[i] - target[j]) * wrong).max(axis=1).mean())


def ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not x.size:
        return [float("nan"), float("nan")]
    q = (1 - level) / 2
    return [float(np.quantile(x, q)), float(np.quantile(x, 1 - q))]


def strat_resample(types, rng, n=None):
    """Indices of a stratified bootstrap sample of scenarios (with replacement, same size per type)."""
    out = []
    for t in np.unique(types):
        members = np.flatnonzero(types == t)
        out.append(rng.choice(members, size=n or len(members), replace=True))
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------- data access
class Data:
    """Planner x scenario matrices for every metric, built once from the tidy run table."""

    def __init__(self, runs, planners=None, seed=0, invalid_as_failure=False):
        self.runs = runs
        main = runs[runs.seed == seed]
        self.planners = planners or sorted(main.planner.unique())
        fam = runs.drop_duplicates("planner").set_index("planner").family
        self.family = np.array([fam[p] for p in self.planners])
        val = main[main.protocol.isin(["ols", "navsim", "clsnr", "clsr"])]
        types = val.drop_duplicates("scenario").set_index("scenario").scenario_type
        self.scenarios = sorted(types.index)
        self.types = np.array([types[s] for s in self.scenarios])
        ip = main[main.protocol == "interplan"]
        self.ip_scenarios = sorted(ip.scenario.unique())
        ip_types = ip.drop_duplicates("scenario").set_index("scenario").scenario_type if len(ip) else pd.Series(dtype=str)
        self.ip_types = np.array([ip_types[s] for s in self.ip_scenarios]) if len(ip) else np.array([])
        self.seed, self.invalid_as_failure = seed, invalid_as_failure
        self._main, self._cache = main, {}

    def matrix(self, protocol, col, sign, scenarios=None):
        """Oriented planner x scenario matrix (NaN = no valid run)."""
        key = (protocol, col, sign)
        if key not in self._cache:
            sc = scenarios or (self.ip_scenarios if protocol == "interplan" else self.scenarios)
            sub = self._main[self._main.protocol == protocol]
            if col not in sub:  # metric not produced in this run (e.g. a protocol not run): all missing
                sc = scenarios or (self.ip_scenarios if protocol == "interplan" else self.scenarios)
                self._cache[key] = np.full((len(self.planners), len(sc)), np.nan)
                return self._cache[key]
            vals = sub[col].astype(float).where(sub.valid.astype(bool))
            if self.invalid_as_failure:
                vals = vals.fillna(WORST.get(col, np.nan) if col in WORST else vals.max())
            piv = pd.DataFrame({"planner": sub.planner, "scenario": sub.scenario, "v": vals}).pivot_table(
                index="planner", columns="scenario", values="v", aggfunc="mean", dropna=False)
            m = piv.reindex(index=self.planners, columns=sc).to_numpy(float)
            self._cache[key] = sign * m
        return self._cache[key]

    def spec(self, name):
        if name in CHEAP:
            return CHEAP[name]
        if name in TARGETS:
            return TARGETS[name]
        if name in REFERENCE:
            return REFERENCE[name]
        if name == "interPlan":
            return INTERPLAN
        if name.startswith("NR-"):  # the same safety outcome measured in the non-reactive closed loop
            p, c, s = TARGETS[name[3:]]
            return ("clsnr", c, s)
        raise KeyError(name)

    def m(self, name):
        return self.matrix(*self.spec(name))


def means(m, idx=None):
    x = m if idx is None else m[:, idx]
    with np.errstate(all="ignore"):
        return np.nanmean(x, axis=1)


def paired_tau(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return tau_b(a[ok], b[ok]) if ok.sum() >= 3 else float("nan")


# ----------------------------------------------------------------------------------------------- RQ1: agreement table
def agreement(d, cheap, target, rng, B=B_MAIN):
    mc, mt = d.m(cheap), d.m(target)
    a, b = means(mc), means(mt)
    ok = ~(np.isnan(a) | np.isnan(b))
    t0, p0 = kendalltau(a[ok], b[ok], variant="b")
    row = {"cheap": cheap, "target": target, "n_planners": int(ok.sum()), "tau": float(t0), "p_value": float(p0),
           "inversion_rate": inversion_rate(a[ok], b[ok]), "mmrv": mmrv(a[ok], b[ok])}
    other_set = target == "interPlan"  # interPlan uses its own scenarios: resample the two sets independently
    A, Bm, AP, BP, AJ, BJ = [], [], [], [], [], []
    for _ in range(B):
        si = strat_resample(d.types, rng)
        ti = strat_resample(d.ip_types, rng) if other_set else si
        a1, b1 = means(mc, si), means(mt, ti)
        A.append(a1), Bm.append(b1)
        pi = rng.choice(np.flatnonzero(ok), size=ok.sum(), replace=True)
        AP.append(a[pi]), BP.append(b[pi])
        AJ.append(a1[pi]), BJ.append(b1[pi])
    for tag, (x, y) in {"scen": (A, Bm), "plan": (AP, BP), "both": (AJ, BJ)}.items():
        x, y = np.array(x), np.array(y)
        if tag == "scen":
            x, y = x[:, ok], y[:, ok]
        good = ~(np.isnan(x).any(1) | np.isnan(y).any(1))
        tb = tau_b(x[good], y[good]) if good.any() else np.array([np.nan])
        row[f"ci_{tag}"] = ci(tb)
        row[f"boot_mean_{tag}"] = float(np.nanmean(tb))
    return row


def available(d, name):
    return not np.isnan(means(d.m(name))).all()


def agreement_table(d, rng, cheap=None, targets=None, B=B_MAIN):
    cheap = [c for c in (cheap or list(CHEAP)) if available(d, c)]
    targets = targets or list(TARGETS) + list(REFERENCE) + (["interPlan"] if len(d.ip_scenarios) else [])
    targets = [t for t in targets if available(d, t)]
    return pd.DataFrame([agreement(d, c, t, rng, B) for c in cheap for t in targets])


def compare_cheap(d, rng, target="AnySafetyViolation", B=B_MAIN):
    """Which cheap score agrees best with a target? Paired differences of tau under the same two-way resamples
    (scenarios stratified by type, and planners), the two-sided bootstrap p-value of each difference, and how often
    each cheap score has the highest tau."""
    names = list(CHEAP)
    mcs, mt = [d.m(c) for c in names], d.m(target)
    full = np.column_stack([means(m) for m in mcs] + [means(mt)])
    ok = ~np.isnan(full).any(1)
    point = np.array([tau_b(full[ok, k], full[ok, -1]) for k in range(len(names))])
    other_set = target == "interPlan"
    boot = np.zeros((B, len(names)))
    for b in range(B):
        si = strat_resample(d.types, rng)
        ti = strat_resample(d.ip_types, rng) if other_set else si
        pi = rng.choice(np.flatnonzero(ok), size=ok.sum(), replace=True)
        tb = means(mt, ti)[pi]
        for k, m in enumerate(mcs):
            boot[b, k] = paired_tau(means(m, si)[pi], tb)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = boot[:, i] - boot[:, j]
            p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
            rows.append({"target": target, "a": names[i], "b": names[j], "tau_a": float(point[i]),
                         "tau_b": float(point[j]), "diff": float(point[i] - point[j]), "ci_diff": ci(diff),
                         "p_boot": float(min(1.0, p))})
    usable = ~np.isnan(boot).all(1)  # resamples in which no cheap score has a defined tau are skipped
    best = (np.bincount(np.nanargmax(boot[usable], axis=1), minlength=len(names)) / usable.sum() if usable.any()
            else np.full(len(names), np.nan))
    return pd.DataFrame(rows), {"target": target, "prob_best": dict(zip(names, map(float, best))),
                                "boot_mean_tau": dict(zip(names, map(float, np.nanmean(boot, 0))))}


# ----------------------------------------------------------------------------------------------- RQ2: slices
def slice_table(d, rng, target="AnySafetyViolation", B=B_SLICE):
    rows = []
    for c in CHEAP:
        mc, mt = d.m(c), d.m(target)
        for t in np.unique(d.types):
            cols = np.flatnonzero(d.types == t)
            a, b = means(mc, cols), means(mt, cols)
            boots = []
            for _ in range(B):
                si = rng.choice(cols, size=len(cols), replace=True)
                boots.append(paired_tau(means(mc, si), means(mt, si)))
            rows.append({"cheap": c, "target": target, "slice": t, "n_scenarios": len(cols), "tau": paired_tau(a, b),
                         "ci_scen": ci(boots), "target_spread": float(np.nanstd(b))})
    return pd.DataFrame(rows)


def reactive_vs_nonreactive(d, rng, B=B_MAIN):
    """Does the cheap-score agreement change when the same safety outcome is measured with reactive agents?"""
    rows = []
    for t in TARGETS:
        mr, mn = d.m(t), d.m("NR-" + t)
        r_vs_nr = [paired_tau(means(mr, si), means(mn, si)) for si in (strat_resample(d.types, rng) for _ in range(B))]
        row = {"target": t, "tau_reactive_vs_nonreactive": paired_tau(means(mr), means(mn)), "ci_scen": ci(r_vs_nr)}
        for c in CHEAP:
            row[f"tau_{c}_R"] = paired_tau(means(d.m(c)), means(mr))
            row[f"tau_{c}_NR"] = paired_tau(means(d.m(c)), means(mn))
        rows.append(row)
    return pd.DataFrame(rows)


def same_vs_cross(d, rng, target="AnySafetyViolation", R=B_SLICE):
    """Protocol effect vs scenario-set effect: tau when cheap score and target use the same half of the scenarios
    against tau when they use disjoint halves (split within every scenario type)."""
    rows = []
    for c in CHEAP:
        mc, mt = d.m(c), d.m(target)
        same, cross = [], []
        for _ in range(R):
            A, Bh = [], []
            for t in np.unique(d.types):
                cols = rng.permutation(np.flatnonzero(d.types == t))
                A.append(cols[: len(cols) // 2]), Bh.append(cols[len(cols) // 2:])
            A, Bh = np.concatenate(A), np.concatenate(Bh)
            same.append(paired_tau(means(mc, A), means(mt, A)))
            cross.append(paired_tau(means(mc, A), means(mt, Bh)))
        same, cross = np.array(same), np.array(cross)
        rows.append({"cheap": c, "target": target, "tau_same_half": float(np.nanmean(same)),
                     "tau_cross_half": float(np.nanmean(cross)), "diff": float(np.nanmean(same - cross)),
                     "ci_diff": ci(same - cross)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- RQ3: composite score
ALPHAS = (0.01, 0.1, 1.0, 10.0)


def _fit_predict(X, y, train, test, alpha):
    mu, sd = X[train].mean(0), X[train].std(0)
    sd[sd == 0] = 1
    model = Ridge(alpha=alpha, positive=True).fit((X[train] - mu) / sd, y[train])
    return model.predict((X[test] - mu) / sd), model.coef_ / sd


def _lofo_predictions(X, y, fam, choose_alpha=True):
    """Out-of-fold predictions with one family held out at a time (alpha chosen by inner leave-one-family-out)."""
    pred, weights = np.full(len(y), np.nan), {}
    for f in np.unique(fam):
        test, train = np.flatnonzero(fam == f), np.flatnonzero(fam != f)
        alpha = 1.0
        if choose_alpha and len(np.unique(fam[train])) >= 3:
            best = -np.inf
            for a in ALPHAS:
                inner = np.full(len(y), np.nan)
                for g in np.unique(fam[train]):
                    te, tr = train[fam[train] == g], train[fam[train] != g]
                    inner[te] = _fit_predict(X, y, tr, te, a)[0]
                score = paired_tau(inner[train], y[train])
                if score > best:
                    best, alpha = score, a
        pred[test], w = _fit_predict(X, y, train, test, alpha)
        weights[f] = {"alpha": alpha, "weights": dict(zip(CHEAP, map(float, w)))}
    return pred, weights


def _best_single_predictions(X, y, fam):
    """Baseline: per held-out family, the single cheap score that ranks the training planners best, mapped to the
    target scale by a one-feature least-squares fit on the training planners."""
    pred, chosen = np.full(len(y), np.nan), {}
    for f in np.unique(fam):
        test, train = np.flatnonzero(fam == f), np.flatnonzero(fam != f)
        taus = [paired_tau(X[train, k], y[train]) for k in range(X.shape[1])]
        if np.all(np.isnan(taus)):  # degenerate fold (too few or tied training planners): nothing to choose from
            chosen[f] = None
            continue
        k = int(np.nanargmax(taus))
        slope, icpt = np.polyfit(X[train, k], y[train], 1)
        pred[test] = slope * X[test, k] + icpt
        chosen[f] = list(CHEAP)[k]
    return pred, chosen


def lofo(d, rng, target="AnySafetyViolation", B=B_LOFO):
    def planner_xy(idx=None):
        X = np.column_stack([means(d.m(c), idx) for c in CHEAP])
        y = means(d.m(target), idx)
        ok = ~(np.isnan(X).any(1) | np.isnan(y))
        return X[ok], y[ok], d.family[ok], np.array(d.planners)[ok]

    X, y, fam, names = planner_xy()
    pred, weights = _lofo_predictions(X, y, fam)
    base, chosen = _best_single_predictions(X, y, fam)
    within = {}
    for f in np.unique(fam):
        sel = fam == f
        if sel.sum() >= 3:
            within[f] = {"composite": paired_tau(pred[sel], y[sel]), "best_single": paired_tau(base[sel], y[sel]),
                         "n": int(sel.sum())}
    rank_true = pd.Series(y).rank(ascending=False).to_numpy()
    rank_pred = pd.Series(pred).rank(ascending=False).to_numpy()
    boots = []
    for _ in range(B):
        Xb, yb, fb, _n = planner_xy(strat_resample(d.types, rng))
        pb, _ = _lofo_predictions(Xb, yb, fb, choose_alpha=False)
        sb, _ = _best_single_predictions(Xb, yb, fb)
        boots.append((paired_tau(pb, yb), paired_tau(sb, yb)))
    boots = np.array(boots)
    raw = {c: paired_tau(X[:, k], y) for k, c in enumerate(CHEAP)}
    return {
        "target": target, "n_planners": int(len(y)), "n_families": int(len(np.unique(fam))),
        "tau_composite_oof": paired_tau(pred, y), "ci_composite": ci(boots[:, 0]),
        "tau_best_single_oof": paired_tau(base, y), "ci_best_single": ci(boots[:, 1]),
        "ci_diff": ci(boots[:, 0] - boots[:, 1]), "best_single_chosen": chosen, "fold_weights": weights,
        "within_family": within, "raw_single_tau": raw,
        "mean_abs_rank_error_composite": float(np.mean(np.abs(rank_pred - rank_true))),
        "planner_predictions": {n: {"family": f, "target": float(t), "pred": float(p)}
                                for n, f, t, p in zip(names, fam, y, pred)},
    }


# ----------------------------------------------------------------------------------------------- RQ4: sensitivity
def sensitivity_invalid(runs, rng, B=B_SLICE):
    base = agreement_table(Data(runs), rng, B=B)
    fail = agreement_table(Data(runs, invalid_as_failure=True), rng, B=B)
    out = base[["cheap", "target", "tau"]].merge(fail[["cheap", "target", "tau"]], on=["cheap", "target"],
                                                 suffixes=("_complete_case", "_invalid_as_failure"))
    out["change"] = out.tau_invalid_as_failure - out.tau_complete_case
    inv = runs[runs.seed == 0].groupby(["planner", "protocol"]).valid.apply(lambda v: float((~v.astype(bool)).mean()))
    return out, inv.rename("invalid_share").reset_index()


def sensitivity_severity(d):
    rows = []
    for col in ("any_violation", "severity_s1", "severity_s2"):
        if col not in d.runs.columns:
            continue
        mt = d.matrix("clsr", col, -1)
        for c in CHEAP:
            rows.append({"cheap": c, "severity_scheme": col, "tau": paired_tau(means(d.m(c)), means(mt))})
    return pd.DataFrame(rows)


def sensitivity_seeds(runs, target="AnySafetyViolation", subset=None):
    """Seeded planners (Diffusion Planner sampling seeds, retrained baselines) on the seed subset: tau over every
    combination of one seed per seeded planner, and each seeded planner's seed-to-seed spread."""
    seeded = sorted(runs[runs.seed > 0].planner.unique())
    if not seeded:
        return pd.DataFrame(), pd.DataFrame()
    subset = subset or sorted(runs[runs.seed > 0].scenario.unique())
    per_seed = {s: Data(runs, seed=s) for s in sorted(runs.seed.unique())}
    d0 = per_seed[0]
    cols0 = [d0.scenarios.index(s) for s in subset if s in d0.scenarios]

    def planner_vec(name, choice):
        v = means(d0.m(name), cols0)
        for p, s in choice.items():
            ds = per_seed[s]
            k = d0.planners.index(p)
            v[k] = means(ds.m(name), [ds.scenarios.index(x) for x in subset if x in ds.scenarios])[ds.planners.index(p)]
        return v

    rows, spread = [], []
    for c in CHEAP:
        taus = []
        for combo in product(sorted(per_seed), repeat=len(seeded)):
            choice = dict(zip(seeded, combo))
            taus.append(paired_tau(planner_vec(c, choice), planner_vec(target, choice)))
        taus = np.array(taus)
        rows.append({"cheap": c, "target": target, "n_combinations": len(taus), "tau_min": float(np.nanmin(taus)),
                     "tau_max": float(np.nanmax(taus)), "tau_sd": float(np.nanstd(taus)),
                     "tau_seed0": float(taus[0])})
    for p in seeded:
        for name in list(CHEAP) + [target]:
            vals = [planner_vec(name, {p: s})[d0.planners.index(p)] for s in sorted(per_seed)]
            spread.append({"planner": p, "metric": name, "values": vals, "sd": float(np.nanstd(vals))})
    return pd.DataFrame(rows), pd.DataFrame(spread)


def sensitivity_count(d, rng, target="AnySafetyViolation", ks=(2, 5, 10, 20, 30, 40, 50), B=B_COUNT):
    rows = []
    mt = d.m(target)
    full_t = means(mt)
    for c in CHEAP:
        mc = d.m(c)
        full_tau = paired_tau(means(mc), full_t)
        for k in ks:
            taus, stab = [], []
            for _ in range(B):
                idx = np.concatenate([rng.choice(np.flatnonzero(d.types == t), size=min(k, (d.types == t).sum()),
                                                 replace=False) for t in np.unique(d.types)])
                taus.append(paired_tau(means(mc, idx), means(mt, idx)))
                stab.append(paired_tau(means(mt, idx), full_t))
            rows.append({"cheap": c, "target": target, "per_type": k, "tau_mean": float(np.nanmean(taus)),
                         "ci": ci(taus), "full_tau": full_tau, "target_rank_stability": float(np.nanmean(stab))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- driver
def planner_table(d):
    rows = []
    for k, p in enumerate(d.planners):
        row = {"planner": p, "family": d.family[k]}
        for name in list(CHEAP) + list(TARGETS) + list(REFERENCE) + (["interPlan"] if len(d.ip_scenarios) else []):
            proto, col, sign = d.spec(name)
            row[name] = float(sign * means(d.m(name))[k])  # back on the natural scale
        rows.append(row)
    return pd.DataFrame(rows)


def run_stats(runs, out_dir, quick=False):
    """Run every analysis; write CSV/JSON tables to out_dir and return the summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    B = 200 if quick else B_MAIN
    d = Data(runs)
    res = {"n_planners": len(d.planners), "n_families": len(np.unique(d.family)), "n_scenarios": len(d.scenarios),
           "n_interplan_scenarios": len(d.ip_scenarios), "bootstrap_B": B}
    pt = planner_table(d)
    pt.to_csv(f"{out_dir}/planner_scores.csv", index=False)
    agr = agreement_table(d, rng, B=B)
    agr.to_csv(f"{out_dir}/agreement.csv", index=False)
    res["agreement"] = agr.to_dict("records")
    # outlying systems can inflate rank correlations (Mathur et al. 2020): repeat without the anchors, and on the
    # Val14 scenarios alone (the 59 top-up scenarios removed)
    keep = [p for p, f in zip(d.planners, d.family) if f.lower() != "anchor"]
    na = agreement_table(Data(runs, planners=keep), rng, B=max(200, B // 4))
    na.to_csv(f"{out_dir}/agreement_no_anchors.csv", index=False)
    res["agreement_no_anchors"] = na.to_dict("records")
    if "source" in runs and (runs.source == "val_topup").any():
        v14 = runs[runs.source != "val_topup"]
        vt = agreement_table(Data(v14), rng, B=max(200, B // 4))
        vt.to_csv(f"{out_dir}/agreement_val14_only.csv", index=False)
        res["agreement_val14_only"] = vt.to_dict("records")
    comp_rows, res["best_cheap"] = [], []
    for t in ["AnySafetyViolation", "AtFaultCollision", "SeverityWeighted"] + (["interPlan"] if len(d.ip_scenarios) else []):
        rows, best = compare_cheap(d, rng, target=t, B=B)
        comp_rows.append(rows)
        res["best_cheap"].append(best)
    comp = pd.concat(comp_rows)
    comp.to_csv(f"{out_dir}/compare_cheap.csv", index=False)
    res["compare_cheap"] = comp.to_dict("records")
    sl = slice_table(d, rng, B=50 if quick else B_SLICE)
    sl.to_csv(f"{out_dir}/slices.csv", index=False)
    res["slices"] = sl.to_dict("records")
    rn = reactive_vs_nonreactive(d, rng, B=B)
    rn.to_csv(f"{out_dir}/reactive_vs_nonreactive.csv", index=False)
    res["reactive_vs_nonreactive"] = rn.to_dict("records")
    sc = same_vs_cross(d, rng, R=50 if quick else B_SLICE)
    sc.to_csv(f"{out_dir}/same_vs_cross.csv", index=False)
    res["same_vs_cross"] = sc.to_dict("records")
    res["lofo"] = {t: lofo(d, rng, target=t, B=20 if quick else B_LOFO)
                   for t in ("AnySafetyViolation", "AtFaultCollision", "SeverityWeighted")}
    inv, inv_share = sensitivity_invalid(runs, rng, B=50 if quick else B_SLICE)
    inv.to_csv(f"{out_dir}/sens_invalid.csv", index=False)
    inv_share.to_csv(f"{out_dir}/invalid_share.csv", index=False)
    res["sens_invalid"] = inv.to_dict("records")
    sev = sensitivity_severity(d)
    sev.to_csv(f"{out_dir}/sens_severity.csv", index=False)
    res["sens_severity"] = sev.to_dict("records")
    seeds, spread = sensitivity_seeds(runs)
    seeds.to_csv(f"{out_dir}/sens_seeds.csv", index=False)
    spread.to_csv(f"{out_dir}/seed_spread.csv", index=False)
    res["sens_seeds"] = seeds.to_dict("records")
    cnt = sensitivity_count(d, rng, B=30 if quick else B_COUNT)
    cnt.to_csv(f"{out_dir}/sens_count.csv", index=False)
    res["sens_count"] = cnt.to_dict("records")
    with open(f"{out_dir}/stats.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--out", default="results/stats")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    r = run_stats(pd.read_csv(a.runs), a.out, quick=a.quick)
    print(json.dumps({k: v for k, v in r.items() if k.startswith("n_")}, indent=1))
