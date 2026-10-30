"""Figures for experiment 2 (PNG, 220 dpi, light theme) from results/stats/*.csv and stats.json.

Colour rules (dataviz reference palette): scatter plots colour planners by three macro-groups (validated all-pairs:
blue, orange, aqua) and mark the seven families with shapes; the agreement heatmap uses the blue<->red diverging scale
with a grey midpoint at tau = 0; text stays in neutral ink. Every figure has a table counterpart in results/stats/.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MACRO = {"Rule-based": SERIES[0], "Released learned": SERIES[1], "Trained here": SERIES[2]}
SHAPES = ["o", "s", "^", "D", "v", "P", "X", "h"]
DIVERGING = LinearSegmentedColormap.from_list(
    "tau", ["#a83232", "#e34948", "#f2b8b0", "#f0efec", "#9ec5f4", "#3987e5", "#1c5cab"])
CHEAP = ["L2", "LogCollision", "OLS", "PDMS", "EPDMS", "CLS-NR"]
CHEAP_LABEL = {"L2": "L2 error", "LogCollision": "Collision with\nlogged agents", "OLS": "nuPlan OLS",
               "PDMS": "NAVSIM PDMS", "EPDMS": "NAVSIM EPDMS", "CLS-NR": "nuPlan CLS-NR"}
TARGET_LABEL = {"AtFaultCollision": "At-fault\ncollision", "TTCViolation": "TTC\nviolation",
                "DrivableAreaViolation": "Drivable-area\nviolation", "DirectionViolation": "Driving-direction\nviolation",
                "AnySafetyViolation": "Any safety\nviolation", "SeverityWeighted": "Severity-\nweighted",
                "CLS-R": "CLS-R score\n(reference)", "interPlan": "interPlan\nscore"}
LOWER_BETTER = {"L2", "LogCollision"}


def parse_ci(v):
    """CI columns are stored as list strings in the CSVs ("[0.1, 0.2]", possibly with nan)."""
    if isinstance(v, str):
        return json.loads(v.replace("nan", "NaN"))
    return v


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
        "axes.titlecolor": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white", "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight"})


def macro_group(planner, family):
    p = planner.lower()
    if p.startswith(("mlp_", "vector_", "raster_")) or family.lower() in ("baseline", "small baselines"):
        return "Trained here"
    if p.startswith(("idm", "log_replay", "const", "pdm_closed")) or family.lower() in ("anchor", "idm"):
        return "Rule-based"
    return "Released learned"


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_agreement(agr, out):
    targets = [t for t in TARGET_LABEL if t in set(agr.target)]
    m = agr.pivot(index="cheap", columns="target", values="tau").reindex(index=CHEAP, columns=targets)
    lo = agr.set_index(["cheap", "target"]).ci_both
    fig, ax = plt.subplots(figsize=(1.05 * len(targets) + 1.6, 4.2))
    ax.grid(False)
    for i, c in enumerate(CHEAP):
        for j, t in enumerate(targets):
            v = m.loc[c, t]
            rgb = DIVERGING((v + 1) / 2)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=rgb))
            ci = lo.get((c, t))
            ci = parse_ci(ci)
            dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.45  # white text on dark cells
            ax.text(j + 0.5, i + 0.42, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if dark else INK, fontweight="bold")
            if ci is not None and len(ci) == 2 and not np.isnan(ci[0]):
                ax.text(j + 0.5, i + 0.72, f"[{ci[0]:.2f}, {ci[1]:.2f}]", ha="center", va="center", fontsize=6.5,
                        color="white" if dark else INK2)
    ax.set_xlim(0, len(targets))
    ax.set_ylim(len(CHEAP), 0)
    ax.set_xticks(np.arange(len(targets)) + 0.5)
    ax.set_xticklabels([TARGET_LABEL[t] for t in targets], fontsize=7.5, color=INK2)
    ax.set_yticks(np.arange(len(CHEAP)) + 0.5)
    ax.set_yticklabels([CHEAP_LABEL[c].replace("\n", " ") for c in CHEAP], color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Kendall tau-b (95% CI, two-way bootstrap)", color=INK2)
    cb.outline.set_visible(False)
    return save(fig, out, "fig2_agreement_heatmap.png")


def fig_scatter(pt, agr, out, target="AnySafetyViolation"):
    fams = list(dict.fromkeys(pt.family))
    shape = {f: SHAPES[k % len(SHAPES)] for k, f in enumerate(fams)}
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.0), sharey=True)
    tau = agr.set_index(["cheap", "target"]).tau
    for ax, c in zip(axes.flat, CHEAP):
        for _, r in pt.iterrows():
            g = macro_group(r.planner, r.family)
            ax.scatter(r[c], 100 * r[target], s=34, marker=shape[r.family], color=MACRO[g], edgecolor="white",
                       linewidth=0.8, zorder=3)
        t = tau.get((c, target), np.nan)
        ax.set_title(f"{CHEAP_LABEL[c].replace(chr(10), ' ')}   tau = {t:.2f}", loc="left")
        ax.set_xlabel(CHEAP_LABEL[c].replace("\n", " ") + (" (lower is better)" if c in LOWER_BETTER else ""))
    for ax in axes[:, 0]:
        ax.set_ylabel("Reactive closed loop: scenarios with\nany safety violation (%)")
    groups = [plt.Line2D([], [], marker="o", ls="", color=col, label=g, markersize=7) for g, col in MACRO.items()]
    families = [plt.Line2D([], [], marker=shape[f], ls="", color=INK2, label=f, markersize=6) for f in fams]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=groups, title="Planner type (colour)", loc="lower left", ncol=3, bbox_to_anchor=(0.06, -0.01),
               fontsize=8, title_fontsize=8)
    fig.legend(handles=families, title="Family (shape)", loc="lower right", ncol=4, bbox_to_anchor=(0.98, -0.01),
               fontsize=8, title_fontsize=8)
    return save(fig, out, "fig3_planner_scatter.png")


def fig_slices(sl, out, show=("CLS-NR", "PDMS", "OLS")):
    sl = sl[sl.cheap.isin(show)].copy()
    order = sl[sl.cheap == show[0]].sort_values("tau").slice.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(order) + 1.2))
    for k, c in enumerate(show):
        d = sl[sl.cheap == c].set_index("slice").reindex(order)
        y = np.arange(len(order)) + (k - 1) * 0.22
        cis = [parse_ci(v) for v in d.ci_scen]
        ax.hlines(y, [ci[0] for ci in cis], [ci[1] for ci in cis], color=SERIES[k], lw=1.4, alpha=0.6)
        ax.plot(d.tau, y, "o", color=SERIES[k], ms=5, label=CHEAP_LABEL[c].replace("\n", " "), zorder=3)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([o.replace("_", " ") for o in order], fontsize=8, color=INK2)
    ax.set_xlabel("Kendall tau-b with the any-safety-violation rate, per scenario type (95% CI over scenarios)")
    lows = [parse_ci(v)[0] for v in sl.ci_scen if parse_ci(v)]
    ax.set_xlim(min([-0.1] + [x - 0.05 for x in lows if x == x]), 1.0)  # the data side of the tau scale only
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    return save(fig, out, "fig4_slices.png")


def fig_reactive(rn, out):
    row = rn.set_index("target").loc["AnySafetyViolation"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for i, c in enumerate(CHEAP):
        a, b = row[f"tau_{c}_NR"], row[f"tau_{c}_R"]
        ax.plot([a, b], [i, i], color=AXIS, lw=2, zorder=1)
        ax.plot(a, i, "o", color=SERIES[0], ms=7, zorder=3, label="Non-reactive agents" if i == 0 else None)
        ax.plot(b, i, "o", color=SERIES[1], ms=7, zorder=3, label="Reactive agents" if i == 0 else None)
    ax.set_yticks(range(len(CHEAP)))
    ax.set_yticklabels([CHEAP_LABEL[c].replace("\n", " ") for c in CHEAP], color=INK2)
    ax.set_xlabel("Kendall tau-b with the any-safety-violation rate")
    ax.legend(loc="lower right", fontsize=8)
    return save(fig, out, "fig5_reactive_vs_nonreactive.png")


def fig_lofo(lofo, out):
    res = lofo["AnySafetyViolation"]
    names = ["Composite (held-out family)", "Best single score (chosen on\ntraining families)"] + \
            [CHEAP_LABEL[c].replace("\n", " ") + " (raw)" for c in CHEAP]
    vals = [res["tau_composite_oof"], res["tau_best_single_oof"]] + [res["raw_single_tau"][c] for c in CHEAP]
    cis = [res["ci_composite"], res["ci_best_single"]] + [None] * len(CHEAP)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(names))[::-1]
    for yi, v, ci, k in zip(y, vals, cis, range(len(names))):
        col = SERIES[0] if k < 2 else MUTED
        ax.barh(yi, v, color=col, height=0.62)
        if ci:
            ax.hlines(yi, ci[0], ci[1], color=INK2, lw=1.2)
        right = max(v, ci[1] if ci else v, 0)
        ax.text(right + 0.02, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=INK2)
    ax.set_xlim(min(0, min(vals) - 0.1), 1.12)
    ax.set_xlabel("Kendall tau-b with the any-safety-violation rate (out-of-family predictions)")
    ax.grid(axis="y", visible=False)
    return save(fig, out, "fig6_lofo_composite.png")


def fig_count(cnt, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ends = []
    for k, c in enumerate(CHEAP):
        d = cnt[cnt.cheap == c].sort_values("per_type")
        cis = [parse_ci(v) for v in d.ci]
        ax.fill_between(d.per_type, [x[0] for x in cis], [x[1] for x in cis], color=SERIES[k], alpha=0.10, lw=0)
        ax.plot(d.per_type, d.tau_mean, "-o", color=SERIES[k], lw=2, ms=4)
        ends.append([d.tau_mean.iloc[-1], d.per_type.iloc[-1], CHEAP_LABEL[c].replace("\n", " ")])
    ends.sort()
    gap = 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i in range(1, len(ends)):  # push labels apart so they do not overlap
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, label in ends:
        ax.text(x + 1, y, label, va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("Scenarios per scenario type (14 types)")
    ax.set_ylabel("Kendall tau-b with the\nany-safety-violation rate")
    ax.set_xlim(0, cnt.per_type.max() + 14)
    return save(fig, out, "fig7_scenario_count.png")


def fig_interplan(pt, agr, out, cheap="CLS-NR"):
    if "interPlan" not in pt:
        return None
    fams = list(dict.fromkeys(pt.family))
    shape = {f: SHAPES[k % len(SHAPES)] for k, f in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for _, r in pt.iterrows():
        ax.scatter(r[cheap], r["interPlan"], s=36, marker=shape[r.family], color=MACRO[macro_group(r.planner, r.family)],
                   edgecolor="white", linewidth=0.8, zorder=3)
    groups = [plt.Line2D([], [], marker="o", ls="", color=col, label=g, markersize=6) for g, col in MACRO.items()]
    families = [plt.Line2D([], [], marker=shape[f], ls="", color=INK2, label=f, markersize=5) for f in fams]
    ax.legend(handles=groups + families, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    t = agr.set_index(["cheap", "target"]).tau.get((cheap, "interPlan"), np.nan)
    ax.set_title(f"tau = {t:.2f}", loc="left")
    ax.set_xlabel(f"{CHEAP_LABEL[cheap].replace(chr(10), ' ')} on the study scenarios")
    ax.set_ylabel("interPlan score (80 long-tail scenarios)")
    return save(fig, out, "fig8_interplan.png")


def make_all(stats_dir, out):
    style()
    agr = pd.read_csv(f"{stats_dir}/agreement.csv")
    for col in ("ci_scen", "ci_plan", "ci_both"):
        agr[col] = agr[col].apply(parse_ci)
    pt = pd.read_csv(f"{stats_dir}/planner_scores.csv")
    st = json.load(open(f"{stats_dir}/stats.json"))
    paths = [fig_agreement(agr, out), fig_scatter(pt, agr, out),
             fig_slices(pd.read_csv(f"{stats_dir}/slices.csv"), out),
             fig_reactive(pd.read_csv(f"{stats_dir}/reactive_vs_nonreactive.csv"), out),
             fig_lofo(st["lofo"], out), fig_count(pd.read_csv(f"{stats_dir}/sens_count.csv"), out),
             fig_interplan(pt, agr, out)]
    return [p for p in paths if p]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/stats")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    for p in make_all(a.stats, a.out):
        print(p)
