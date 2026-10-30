#!/bin/bash
# Installation check for the experiment-2 harness on a simulation host (Linux, environments built from env/, data and
# checkpoints downloaded): nine planners on two scenarios under the three nuPlan protocols plus one interPlan scenario,
# then metric extraction and NAVSIM scoring, all inside this repository (folders repro/, repro_results/).
# Usage, from the repository root:  EXP2_ROOT=$(pwd) bash tools/repro_smoke.sh      (about 5 minutes on 32 vCPUs)
set -euo pipefail
R=${EXP2_ROOT:?set EXP2_ROOT to the repository root}
cd "$R"
J=$R/repro_jobs.jsonl
OUT=$R/repro_results
mkdir -p "$OUT"
ONLY=__idm_default__,__pdm_closed__,__pdm_open__,__gc_pgp__,__plantf__,__pluto__,__dp_default__,__mlp_final__,__log_replay__
source "$R/env/env.sh" nuplan > /dev/null
export PYTHONPATH=$R/code:${PYTHONPATH:-}
python code/runner.py make --registry code/planners_core.yaml,code/planners_learned.yaml,code/planners_baselines.yaml \
  --limit 2 --tag repro --jobs "$J"
python code/runner.py run --workers "$(( $(nproc --all) > 32 ? 32 : $(nproc --all) ))" --retries 0 --jobs "$J" --only "$ONLY"
python - "$J" "$ONLY" > "$J.selected" <<'PY'
import json, sys
only = sys.argv[2].split(",")
for line in open(sys.argv[1]):
    if any(s in json.loads(line)["id"] for s in only):
        print(line, end="")
PY
python code/extract_metrics.py --jobs "$J.selected" --out "$OUT/runs.csv"
source "$R/env/env.sh" navsim > /dev/null
export PYTHONPATH=$R/code:${PYTHONPATH:-}
python code/navsim_score.py --jobs "$J.selected" --out "$OUT/navsim_scores.csv" --workers 4
python - "$OUT" <<'PY'
import sys
import pandas as pd
r = pd.read_csv(sys.argv[1] + "/runs.csv")
n = pd.read_csv(sys.argv[1] + "/navsim_scores.csv")
print(r.groupby(["planner", "protocol"]).valid.agg(["size", "sum"]).unstack().to_string())
print("NAVSIM rows", len(n), "valid", int(n.valid.sum()))
ok = r.valid.all() and n.valid.all() and len(n) > 0
print("REPRO SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
