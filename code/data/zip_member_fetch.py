"""Fetch members of a remote zip (public S3 / HTTPS) with HTTP range requests, without downloading the whole archive.

Usage:
  python zip_member_fetch.py URL OUT_DIR (--names-file FILE | --all) [--workers N] [--list] [--probe]

FILE holds one name per line: a log name, a basename such as <log>.db, or a full member path.
Each selected member is streamed with one ranged GET, inflated on the fly, checked against the CRC-32 and size in
the central directory, and written as OUT_DIR/<basename> via a .part file. Members whose output already exists
with the right size are skipped, so the command can simply be re-run after an interruption.
--list prints the selection without downloading; --probe reads only the first 64 KiB of each member.
"""
import argparse
import concurrent.futures as cf
import io
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

LOCAL_SIG = 0x04034B50
CHUNK = 8 << 20


def stem(name):
    return name[:-3] if name.endswith(".db") else name


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        with ranged(self.url, self.pos, end) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def ranged(url, start, end, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"}), timeout=timeout)


def data_offset(url, info):
    with ranged(url, info.header_offset, info.header_offset + 29) as r:
        h = r.read()
    sig, _, _, method, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", h)
    if sig != LOCAL_SIG or method != info.compress_type:
        raise ValueError(f"bad local header for {info.filename}")
    return info.header_offset + 30 + nlen + xlen


def fetch(url, info, out_dir, probe):
    out = os.path.join(out_dir, os.path.basename(info.filename))
    if not probe and os.path.exists(out) and os.path.getsize(out) == info.file_size:
        return "skip", 0
    start = data_offset(url, info)
    end = start + info.compress_size - 1
    if probe:
        end = min(end, start + 65535)
    inflater = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    with ranged(url, start, end) as r:
        if probe:
            raw = r.read()
            head = inflater.decompress(raw) if inflater else raw
            return "probe", head[:16]
        crc, n, tmp = 0, 0, out + ".part"
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                data = inflater.decompress(chunk) if inflater else chunk
                crc, n = zlib.crc32(data, crc), n + len(data)
                f.write(data)
            if inflater:
                data = inflater.flush()
                crc, n = zlib.crc32(data, crc), n + len(data)
                f.write(data)
    if n != info.file_size or crc != info.CRC:
        os.remove(tmp)
        raise ValueError(f"size/CRC mismatch for {info.filename}: {n} vs {info.file_size}")
    os.replace(tmp, out)
    return "ok", info.compress_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out_dir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--names-file")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    infos = [i for i in zipfile.ZipFile(HttpRangeFile(a.url)).infolist() if not i.is_dir()]
    if a.names_file:
        wanted = {w.strip() for w in open(a.names_file, encoding="utf-8") if w.strip()}
        def hit(i):
            b = os.path.basename(i.filename)
            return i.filename in wanted or b in wanted or stem(b) in wanted
        infos = [i for i in infos if hit(i)]
        found = {stem(os.path.basename(i.filename)) for i in infos}
        missing = sorted(w for w in wanted if stem(w.rsplit("/", 1)[-1]) not in found)
        if missing:
            print("NOT IN ARCHIVE:", ", ".join(missing))
    tc, tu = sum(i.compress_size for i in infos), sum(i.file_size for i in infos)
    print(f"selected {len(infos)} members: {tc / 1e9:.2f} GB compressed, {tu / 1e9:.2f} GB extracted")
    if a.list:
        for i in infos:
            print(f"   {i.filename}  {i.compress_size}  {i.file_size}")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)

    def job(i):
        for k in range(a.retries + 1):
            try:
                return i, fetch(a.url, i, a.out_dir, a.probe)
            except Exception as e:
                if k == a.retries:
                    return i, ("FAILED", repr(e))
                time.sleep(5 * (k + 1))

    t0, done, got, failed = time.time(), 0, 0, []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, (status, val) in ex.map(job, infos):
            done += 1
            if status == "FAILED":
                failed.append(i.filename)
            if status == "ok":
                got += val
            rate = got / max(time.time() - t0, 1e-6) / 1e6
            print(f"[{done}/{len(infos)}] {status} {os.path.basename(i.filename)} {val if status in ('probe', 'FAILED') else ''} {rate:.0f} MB/s", flush=True)
    print(f"finished in {time.time() - t0:.0f} s; failed {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
