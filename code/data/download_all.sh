#!/bin/bash
# Download everything experiment 2 needs into /data (about 130 GB to download, 220 GB on disk; CRC-checked, resumable:
# re-running skips finished files). Needs python3, aria2c, curl, unzip and the AWS CLI (for the public S3 maps zip).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
S3=https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1
D=/data/nuplan/dataset/nuplan-v1.1/splits
L=/data/.fetch
mkdir -p "$L" "$D"
log() { echo "$(date -u +%FT%TZ) $*" >> "$L/driver.log"; }
zipget() {
  local name=$1 url=$2 out=$3; shift 3
  for attempt in 1 2 3; do
    log "START $name attempt $attempt"
    python3 "$HERE/zip_member_fetch.py" "$url" "$out" "$@" > "$L/$name.out" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then log "DONE $name"; return 0; fi
    log "RETRY $name exit=$rc"
  done
  log "FAILED $name after 3 attempts"
  return 1
}
log "BEGIN"
python3 "$HERE/fetch_files.py" "$HERE/small_files_manifest.json" --root /data --jobs 3 > "$L/small.out" 2>&1
log "small manifest exit=$?"
zipget test "$S3/nuplan-v1.1_test.zip" "$D/test" --names-file "$HERE/interplan10_logs.txt" --workers 7
zipget val "$S3/nuplan-v1.1_val.zip" "$D/val" --all --workers 16
zipget train_pittsburgh "$S3/nuplan-v1.1_train_pittsburgh.zip" "$D/train_pittsburgh" --all --workers 12
log "ALL_DONE"
