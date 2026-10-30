"""Merge the run tables and collect the results of experiment 2 into results/summary.json.

  python code/summarize.py merge      # results/runs_merged.csv = runs.csv + NAVSIM rows (protocol "navsim")
  python code/summarize.py summary    # results/summary.json from runs_merged.csv, stats/, scenarios/, registries, logs
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CODE = os.path.join(ROOT, "code")
SCEN = next((d for d in (os.path.join(RES, "scenarios"), os.path.join(ROOT, "scenarios")) if os.path.isdir(d)),
            os.path.join(RES, "scenarios"))  # the host writes results/scenarios; this repository ships scenarios/
REGISTRIES = ["planners_core.yaml", "planners_learned.yaml", "planners_baselines.yaml"]
FAMILY_LABEL = {"IDM": "IDM", "PDM": "PDM", "Released imitation": "Released imitation", "PLUTO": "PLUTO",
                "Diffusion Planner": "Diffusion Planner", "Baseline": "Small baselines (trained here)", "Anchor": "Anchors"}
NAVSIM_PUBLISHED_HUMAN_PDMS = 94.8  # NAVSIM v1 paper, Table 1, human agent on navtest
NAVSIM_PUBLISHED_HUMAN_SUB = {"NC": 100.0, "DAC": 100.0, "TTC": 100.0, "C": 99.9, "EP": 87.5}  # same row
SUB_COLS = {"NC": "nc_raw", "DAC": "dac_raw", "TTC": "ttc_raw", "C": "comfort_v1", "EP": "ep_raw"}


def registry():
    reg = {}
    for f in REGISTRIES:
        reg.update(yaml.safe_load(open(os.path.join(CODE, f)))["planners"])
    return reg


def merge():
    runs = pd.read_csv(os.path.join(RES, "runs.csv"), low_memory=False)
    nav_path = os.path.join(RES, "navsim_scores.csv")
    if os.path.exists(nav_path):
        nav = pd.read_csv(nav_path, low_memory=False)
        meta = runs[runs.protocol == "ols"].drop_duplicates("scenario").set_index("scenario")
        for c in ("scenario_type", "log_name", "source"):
            nav[c] = nav.scenario.map(meta[c]) if c in meta else None
        runs = pd.concat([runs[runs.protocol != "navsim"], nav], ignore_index=True)
    runs.to_csv(os.path.join(RES, "runs_merged.csv"), index=False)
    print(runs.groupby("protocol").valid.agg(["size", "sum"]))


def _pct(x):
    return float(100 * x)


def summary():
    runs = pd.read_csv(os.path.join(RES, "runs_merged.csv"), low_memory=False)
    st = json.load(open(os.path.join(RES, "stats", "stats.json")))
    sel = json.load(open(os.path.join(SCEN, "selection.json")))
    reg = registry()
    main = runs[runs.seed == 0]
    planners = sorted(main.planner.unique())
    fam = {}
    for p in planners:
        fam.setdefault(reg.get(p, {}).get("family", main[main.planner == p].family.iloc[0]), []).append(p)
    order = ["IDM", "PDM", "Released imitation", "PLUTO", "Diffusion Planner", "Baseline", "Anchor"]
    fam = {FAMILY_LABEL.get(f, f): sorted(fam[f]) for f in sorted(fam, key=lambda f: order.index(f) if f in order else 99)}
    val = pd.read_csv(os.path.join(SCEN, "val700.csv"))
    final_counts = val.scenario_type.value_counts().to_dict()
    S = {"design": {
        "n_planners": len(planners), "n_families": len(fam), "families": fam,
        "n_scenarios": int(val.token.nunique()), "n_val14": int((val.source == "val14").sum()),
        "n_topup": int((val.source == "val_topup").sum()), "n_logs": int(val.log_name.nunique()),
        "maps": val.map_name.value_counts().to_dict(), "per_type": final_counts,
        "n_seed": int(pd.read_csv(os.path.join(SCEN, "seed140.csv")).token.nunique()),
        "n_interplan": int(main[main.protocol == "interplan"].scenario.nunique()),
        "val14_size": int(sel["val14_resolved"]),
        "seeded_planners": sorted(runs[runs.seed > 0].planner.unique()),
    }}
    # runs, validity and compute
    by = runs.groupby("protocol")
    S["runs"] = {"by_protocol": {p: {"runs": int(len(g)), "valid": int(g.valid.astype(bool).sum())} for p, g in by},
                 "total": int(len(runs[runs.protocol != "navsim"])),
                 "total_valid": int(runs[runs.protocol != "navsim"].valid.astype(bool).sum()),
                 "sim_hours": float(runs.get("sim_duration_s", pd.Series(dtype=float)).sum() / 3600)}
    inv = main[main.protocol.isin(["ols", "clsnr", "clsr", "interplan"])].groupby("planner").valid.apply(
        lambda v: float(1 - v.astype(bool).mean()))
    S["runs"]["invalid_share_by_planner"] = inv.round(4).to_dict()
    ok = runs[runs.valid.astype(bool)]
    if "sim_duration_s" in ok:  # cost of each protocol per scenario (runner report wall time, one core)
        S["runs"]["mean_sim_s_by_protocol"] = ok.groupby("protocol").sim_duration_s.mean().round(2).dropna().to_dict()
    if "planner_runtime_s" in ok:
        S["runs"]["planner_ms_per_step"] = (ok[ok.protocol == "clsr"].groupby("planner").planner_runtime_s.mean()
                                            * 1000).round(1).to_dict()
    errs = main[~main.valid.astype(bool)].get("error")
    S["runs"]["top_errors"] = errs.fillna("").str[:120].value_counts().head(8).to_dict() if errs is not None else {}
    # causes of invalid simulations (all seeds; NAVSIM rows have their own validity), largest first
    bad = runs[~runs.valid.astype(bool) & (runs.protocol != "navsim")]
    causes = []
    for err, g in bad.groupby(bad.get("error", pd.Series("unknown", index=bad.index)).fillna("unknown")):
        causes.append({"error": str(err)[:200], "n": int(len(g)),
                       "planners": {k: int(v) for k, v in g.planner.value_counts().items()},
                       "protocols": {k: int(v) for k, v in g.protocol.value_counts().items()}})
    S["runs"]["invalid_causes"] = sorted(causes, key=lambda c: -c["n"])[:8]
    # planner table (natural-scale planner means) and validation anchors
    pt = pd.read_csv(os.path.join(RES, "stats", "planner_scores.csv"))
    S["planner_scores"] = pt.round(4).to_dict("records")
    nav = main[(main.protocol == "navsim") & main.valid.astype(bool)]
    S["validation"] = {"human_pdms": _pct(nav[nav.planner == "log_replay"].pdms.mean()) if len(nav) else None,
                       "human_epdms": _pct(nav[nav.planner == "log_replay"].epdms.mean()) if len(nav) else None,
                       "navsim_published_human_pdms": NAVSIM_PUBLISHED_HUMAN_PDMS,
                       "navsim_published_human_sub": NAVSIM_PUBLISHED_HUMAN_SUB}
    if len(nav):  # navtest's selection rule: keep frames where the human scores >= 0.8 and constant velocity <= 0.8
        h = nav[nav.planner == "log_replay"].set_index("scenario")
        cv = nav[nav.planner == "const_velocity"].set_index("scenario")
        keep = h.index[(h.pdms >= 0.8) & (cv.pdms.reindex(h.index) <= 0.8)]
        S["validation"].update(
            human_sub_all={k: _pct(h[c].mean()) for k, c in SUB_COLS.items() if c in h},
            navtest_rule_n=int(len(keep)), human_pdms_navtest_rule=_pct(h.loc[keep].pdms.mean()),
            human_sub_navtest_rule={k: _pct(h.loc[keep, c].mean()) for k, c in SUB_COLS.items() if c in h})
    # statistics (copied, plus convenient highlights)
    for k in ("agreement", "agreement_no_anchors", "agreement_val14_only", "compare_cheap", "best_cheap", "slices",
              "reactive_vs_nonreactive", "same_vs_cross", "lofo", "sens_invalid", "sens_severity", "sens_seeds",
              "sens_count", "bootstrap_B"):
        if k in st:
            S[k] = st[k]
    agr = pd.DataFrame(st["agreement"])
    hi = {}
    for t in agr.target.unique():
        a = agr[agr.target == t].sort_values("tau", ascending=False)
        hi[t] = {"best": a.iloc[0].cheap, "best_tau": float(a.iloc[0].tau), "best_ci": a.iloc[0].ci_both,
                 "worst": a.iloc[-1].cheap, "worst_tau": float(a.iloc[-1].tau),
                 "taus": dict(zip(a.cheap, a.tau.round(4)))}
    S["highlights"] = hi
    # training of the small baselines and compute log
    mp = os.path.join(RES, "model_params.json")  # code/count_params.py on the host
    if os.path.exists(mp):
        S["model_params"] = json.load(open(mp))
    man = os.path.join(RES, "baselines", "manifest.json")
    if os.path.exists(man):
        S["baselines_training"] = json.load(open(man))
    log = os.path.join(RES, "logs", "bigrun.log")
    if os.path.exists(log):
        stamps = {m.group(2): m.group(1) for m in re.finditer(r"^(\S+Z) (.+)$", open(log).read(), re.M)}
        S["bigrun_log"] = stamps

        def at(prefix):
            return next((pd.Timestamp(v) for k, v in stamps.items() if k.startswith(prefix)), None)
        start, end, done = at("full pass start"), at("full pass exit"), at("BIGRUN_DONE")
        workers = next((int(m.group(1)) for k in stamps for m in [re.search(r"with (\d+) workers", k)] if m), None)
        if start is not None and end is not None:
            S["compute"] = {"pass_hours": round((end - start).total_seconds() / 3600, 2), "workers": workers,
                            "scoring_hours": round((done - end).total_seconds() / 3600, 2) if done is not None else None,
                            "host": "AWS EC2 m8a.48xlarge (192 vCPU, 768 GiB)"}
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("summary.json written:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in S.items()})


if __name__ == "__main__":
    {"merge": merge, "summary": summary}[sys.argv[1] if len(sys.argv) > 1 else "summary"]()
