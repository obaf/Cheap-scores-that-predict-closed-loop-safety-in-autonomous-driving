#!/usr/bin/env python3
"""Fetch the experiment-2 data on the EC2 host from a manifest (JSON list of items).

Resumable: an item with a .done marker is skipped, and aria2c resumes partial downloads. Each item is downloaded,
checked against its expected size, extracted into place under the data root, and its archive deleted. A failed item
is logged to <root>/.fetch/<id>.failed and the remaining items carry on.

Item fields: id, url, method (https | s3-nosign | curl | git), bytes (expected size, optional), archive
(zip | tgz | tar | none), extract_to (path under root), strip (optional sub-folder inside the archive whose contents
go to extract_to), filename (optional), region (s3 only), ref (git only), lfs (git only).

Usage: python3 exp2_fetch.py manifest.json [--root /data] [--jobs 3] [--only id1,id2] [--dry-run]
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

LOCK = threading.Lock()
LOG = None


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with LOCK:
        print(line, flush=True)
        with open(LOG, "a") as f:
            print(line, file=f)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()[-800:]
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}: {tail}")
    return r


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def merge(src, dest):
    """Move src (file or folder) to dest, merging folders that already exist."""
    if not dest.exists():
        os.rename(src, dest)
        return
    if src.is_dir() and dest.is_dir():
        for child in src.iterdir():
            merge(child, dest / child.name)
        src.rmdir()
    else:
        os.replace(src, dest)


def download(item, dl):
    url = item["url"]
    name = item.get("filename") or url.split("?")[0].rstrip("/").split("/")[-1]
    out = dl / name
    method = item["method"]
    if method == "s3-nosign":
        run(["aws", "s3", "cp", url, str(out), "--no-sign-request", "--only-show-errors",
             "--region", item.get("region", "ap-northeast-1")])
    elif method == "https":
        run(["aria2c", "-q", "-c", "-x16", "-s16", "-k16M", "--file-allocation=none", "--max-tries=10",
             "--retry-wait=10", "--auto-file-renaming=false", "--allow-overwrite=true", "-d", str(dl), "-o", name, url])
    elif method == "curl":
        run(["curl", "-fsSL", "--retry", "10", "-C", "-", "-o", str(out), url])
    else:
        raise ValueError(f"unknown method {method}")
    size = out.stat().st_size
    if item.get("bytes") and size != item["bytes"]:
        raise RuntimeError(f"size {size:,} != expected {item['bytes']:,}")
    return out, size


def extract(item, archive, root):
    target = root / item["extract_to"]
    target.mkdir(parents=True, exist_ok=True)
    kind = item.get("archive", "none")
    if kind == "none":
        os.replace(archive, target / archive.name)
        return
    tmp = root / "tmp_extract" / item["id"]
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if kind == "zip":
        run(["unzip", "-q", "-o", str(archive), "-d", str(tmp)])
    elif kind == "tgz":
        run(["tar", "-I", "pigz", "-xf", str(archive), "-C", str(tmp)])
    elif kind == "tar":
        run(["tar", "-xf", str(archive), "-C", str(tmp)])
    else:
        raise ValueError(f"unknown archive type {kind}")
    src = tmp / item["strip"] if item.get("strip") else tmp
    if not src.is_dir():
        raise RuntimeError(f"strip folder {item.get('strip')} not found; archive top level: {sorted(p.name for p in tmp.iterdir())[:10]}")
    for child in list(src.iterdir()):
        merge(child, target / child.name)
    shutil.rmtree(tmp, ignore_errors=True)
    archive.unlink()


def git_item(item, root):
    target = root / item["extract_to"]
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", item["url"], str(target)])
    if item.get("ref"):
        run(["git", "-C", str(target), "checkout", "--quiet", item["ref"]])
    if item.get("lfs"):
        run(["git", "-C", str(target), "lfs", "pull"])
    return run(["git", "-C", str(target), "rev-parse", "HEAD"]).stdout.strip()


def du_bytes(path):
    r = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True)
    return int(r.stdout.split()[0]) if r.returncode == 0 and r.stdout else None


def fetch(item, root, state, dl):
    iid = item["id"]
    done = state / f"{iid}.done"
    if done.exists():
        log(f"SKIP {iid}: already done")
        return iid, "skipped"
    (state / f"{iid}.failed").unlink(missing_ok=True)
    t0 = time.time()
    try:
        info = {"id": iid, "url": item["url"]}
        if item["method"] == "git":
            log(f"START {iid}: git clone {item['url']}")
            info["commit"] = git_item(item, root)
        else:
            need = (item.get("bytes") or 0) * 2.2 / 1e9
            if need and free_gb(root) < need:
                raise RuntimeError(f"only {free_gb(root):.0f} GB free, need about {need:.0f} GB")
            log(f"START {iid}: {(item.get('bytes') or 0) / 1e9:.2f} GB via {item['method']}")
            archive, size = download(item, dl)
            info["downloaded_bytes"] = size
            t1 = time.time()
            log(f"GOT {iid}: {size / 1e9:.2f} GB in {t1 - t0:.0f} s ({size / 1e6 / max(t1 - t0, 1):.0f} MB/s); extracting")
            extract(item, archive, root)
        info["seconds"] = round(time.time() - t0)
        info["installed_bytes"] = du_bytes(root / item["extract_to"])
        done.write_text(json.dumps(info, indent=1))
        log(f"DONE {iid} in {info['seconds']} s; {item['extract_to']} now {(info['installed_bytes'] or 0) / 1e9:.2f} GB")
        return iid, "done"
    except Exception as e:
        (state / f"{iid}.failed").write_text(str(e))
        log(f"FAIL {iid}: {e}")
        return iid, "failed"


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--root", default="/data")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    state, dl = root / ".fetch", root / "downloads"
    state.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    LOG = state / "fetch.log"
    items = json.loads(Path(a.manifest).read_text())
    if a.only:
        keep = set(a.only.split(","))
        items = [i for i in items if i["id"] in keep]
    total = sum(i.get("bytes") or 0 for i in items)
    log(f"PLAN {len(items)} items, {total / 1e9:.1f} GB to download, {free_gb(root):.0f} GB free on {root}")
    if a.dry_run:
        for i in items:
            log(f"  {i['id']}: {i['method']} {(i.get('bytes') or 0) / 1e9:.2f} GB -> {i['extract_to']}")
        return
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(lambda i: fetch(i, root, state, dl), items))
    summary = {s: [i for i, r in results if r == s] for s in ("done", "skipped", "failed")}
    (state / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    log(f"FINISHED done={len(summary['done'])} skipped={len(summary['skipped'])} failed={summary['failed']}")


if __name__ == "__main__":
    main()
