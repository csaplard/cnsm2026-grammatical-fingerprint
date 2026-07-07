"""Data acquisition for the CNSM 2026 grammatical-fingerprint study.

Downloads the Telecom Italia Big Data Challenge Milano telecommunications
dataset (Harvard Dataverse, doi:10.7910/DVN/EGZHFV, v1.3, ODbL 1.0), one
daily file at a time. Each file is MD5-verified against the Dataverse
metadata, reduced to per-cell 10-minute activity matrices (summed over
country codes, all five channels kept), stored as data/daily/<date>.npz,
and the ~330 MB raw file is deleted. Provenance goes to data/manifest.json.

Resumable: days whose .npz already exists and whose manifest entry says
"verified" are skipped.

The dataset is protected by a Dataverse guestbook ("Privacy risk
assessment", id 96) that requires an email address per file download. The
email is supplied explicitly via --email; the script POSTs it as the
guestbook response and downloads through the signed URL the API returns.

Usage: python src/acquire.py --email ADDRESS [--scratch DIR]
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DOI = "doi:10.7910/DVN/EGZHFV"
API = "https://dataverse.harvard.edu/api"
UA = {"User-Agent": "cnsm2026-fingerprint-study/1.0 (research data acquisition)"}
CHANNELS = ["sms_in", "sms_out", "call_in", "call_out", "internet"]
COLUMNS = ["square_id", "time_interval", "country_code"] + CHANNELS
N_SQUARES = 10000
INTERVAL_MS = 600_000  # 10 minutes

ROOT = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT / "data" / "daily"
MANIFEST = ROOT / "data" / "manifest.json"


def api_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return json.load(r)


def list_files():
    meta = api_json(f"{API}/datasets/:persistentId?persistentId={DOI}")
    v = meta["data"]["latestVersion"]
    files = []
    for f in v["files"]:
        df = f["dataFile"]
        files.append({
            "id": df["id"],
            "filename": df["filename"],
            "filesize": df["filesize"],
            "md5": df["md5"],
        })
    files.sort(key=lambda x: x["filename"])
    version = f"{v.get('versionNumber')}.{v.get('versionMinorNumber')}"
    return version, files


def signed_url(file_id, email):
    """Satisfy the guestbook requirement; returns a one-time download URL."""
    body = json.dumps({"guestbookResponse": {"email": email}}).encode()
    req = urllib.request.Request(
        f"{API}/access/datafile/{file_id}", data=body, method="POST",
        headers={**UA, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    data = resp.get("data", {})
    for key in ("url", "signedUrl", "downloadUrl"):
        if isinstance(data.get(key), str):
            return data[key]
    if isinstance(data.get("signedUrls"), list) and data["signedUrls"]:
        first = data["signedUrls"][0]
        return first if isinstance(first, str) else first.get("signedUrl") or first.get("url")
    raise IOError(f"unrecognized guestbook response schema: {json.dumps(resp)[:400]}")


def download(file_id, dest, expected_size, email):
    url = signed_url(file_id, email)
    md5 = hashlib.md5()
    got = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, \
            open(dest, "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            md5.update(chunk)
            out.write(chunk)
            got += len(chunk)
    if got != expected_size:
        raise IOError(f"size mismatch: got {got}, expected {expected_size}")
    return md5.hexdigest()


def reduce_day(raw_path, date_str):
    """Aggregate one daily file to per-(square, interval) sums over country codes."""
    df = pd.read_csv(raw_path, sep="\t", names=COLUMNS,
                     dtype={"square_id": np.int32, "time_interval": np.int64,
                            "country_code": np.int32},
                     na_values=[""], engine="c")
    day_start = pd.Timestamp(date_str, tz="Europe/Rome").tz_convert("UTC")
    t0 = int(day_start.timestamp() * 1000)
    col = ((df["time_interval"].to_numpy() - t0) // INTERVAL_MS).astype(np.int64)
    n_cols = int(col.max()) + 1
    row = df["square_id"].to_numpy() - 1  # squares are 1-based
    if row.min() < 0 or row.max() >= N_SQUARES or col.min() < 0:
        raise ValueError(f"{date_str}: index out of range (rows {row.min()}..{row.max()}, cols {col.min()}..)")
    flat = row * n_cols + col
    mats = {}
    for ch in CHANNELS:
        vals = df[ch].to_numpy(dtype=np.float64)
        mask = ~np.isnan(vals)
        sums = np.bincount(flat[mask], weights=vals[mask], minlength=N_SQUARES * n_cols)
        counts = np.bincount(flat[mask], minlength=N_SQUARES * n_cols)
        m = sums.reshape(N_SQUARES, n_cols).astype(np.float32)
        m[counts.reshape(N_SQUARES, n_cols) == 0] = np.nan
        mats[ch] = m
    return t0, n_cols, mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True,
                    help="email for the Dataverse guestbook response (required by the dataset)")
    ap.add_argument("--scratch", default=None, help="directory for temporary raw downloads")
    args = ap.parse_args()
    scratch = Path(args.scratch) if args.scratch else DAILY_DIR.parent / "raw"
    scratch.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    version, files = list_files()
    print(f"dataset version {version}, {len(files)} files", flush=True)

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    manifest.setdefault("dataset", {"doi": DOI, "version": version, "license": "ODbL 1.0"})
    manifest.setdefault("files", {})

    for f in files:
        name = f["filename"]                      # sms-call-internet-mi-YYYY-MM-DD.txt
        date_str = name.replace("sms-call-internet-mi-", "").replace(".txt", "")
        out_npz = DAILY_DIR / f"{date_str}.npz"
        entry = manifest["files"].get(name, {})
        if out_npz.exists() and entry.get("status") == "verified":
            print(f"{date_str}: already done, skipping", flush=True)
            continue

        t_start = time.time()
        raw = scratch / name
        print(f"{date_str}: downloading {f['filesize']/1e6:.0f} MB ...", flush=True)
        actual_md5 = download(f["id"], raw, f["filesize"], args.email)
        if actual_md5 != f["md5"]:
            raw.unlink(missing_ok=True)
            raise IOError(f"{name}: MD5 mismatch (expected {f['md5']}, got {actual_md5})")

        t0, n_cols, mats = reduce_day(raw, date_str)
        np.savez_compressed(out_npz, t0_ms=np.int64(t0), n_intervals=np.int64(n_cols),
                            **{ch: mats[ch] for ch in CHANNELS})
        raw.unlink()

        manifest["files"][name] = {
            "dataverse_file_id": f["id"], "filesize": f["filesize"],
            "md5_expected": f["md5"], "md5_actual": actual_md5,
            "status": "verified", "n_intervals": n_cols,
            "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=1))
        print(f"{date_str}: done in {time.time()-t_start:.0f}s ({n_cols} intervals)", flush=True)

    print("acquisition complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
