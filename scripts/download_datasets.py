#!/usr/bin/env python3
"""
RailScan - download the raw datasets and build the CNN image metadata.

1. Signboard images (CC BY 4.0)
   https://huggingface.co/datasets/jai23023/indicstr12-signboards
   -> data/raw/signboards/{train,validation}-00000-of-00001.parquet (SHA-256 checked)
   -> data/sign_images/<Script>/<script>_<sha1[:12]>.<ext>
   -> data/processed/RailScan_CNN_Metadata.csv

2. Railway running records (GPL-3.0), one CSV per train
   https://huggingface.co/datasets/yadavsaurabh/indian-railways-runs
   -> data/raw/delay/<train_no>.csv  (only the trains used by RailScan)

Downloads resume after a broken connection and are skipped when the file
already exists with the right checksum / size. Nothing is generated: every
image and row comes from the source files.

Usage:  python scripts/download_datasets.py [--skip-delay] [--skip-signboards]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIGN_REPO = "jai23023/indicstr12-signboards"
DELAY_REPO = "yadavsaurabh/indian-railways-runs"
HF = "https://huggingface.co/datasets"
# Git-LFS SHA-256 of the parquet files (from the Hugging Face file listing).
SIGN_FILES = {
    "train-00000-of-00001.parquet": "009143f6b57297b67c7b22592de83ffcb8cf32ad8bd8c236ed2f1c03b16ecd34",
    "validation-00000-of-00001.parquet": "d3bb7f76758eebe2c7aa4fb697159c8339ddc78149713cce1e9829adcb8ec528",
}
DELAY_TRAINS = ["01023", "01024"]
# Hindi and Marathi are both written in Devanagari; the CNN classifies scripts.
LANGUAGE_TO_SCRIPT = {"hindi": "Devanagari", "marathi": "Devanagari",
                      "kannada": "Kannada", "telugu": "Telugu"}
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SEED = 42


# ------------------------------------------------------------------ download
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, attempts: int = 8) -> None:
    """Download with HTTP range resume; retries on connection errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(dest, mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416:          # range not satisfiable: already complete
                return
            print(f"  attempt {i}: HTTP {exc.code}")
        except OSError as exc:
            print(f"  attempt {i}: {exc}")
        time.sleep(min(2 ** i, 30))
    raise SystemExit(f"Download failed after {attempts} attempts: {url}")


def fetch_signboards(raw_dir: Path) -> None:
    for name, digest in SIGN_FILES.items():
        dest = raw_dir / name
        if dest.exists() and sha256(dest) == digest:
            print(f"  ok (cached)  {dest.relative_to(ROOT)}")
            continue
        print(f"  downloading  {name}")
        download(f"{HF}/{SIGN_REPO}/resolve/main/data/{name}", dest)
        if sha256(dest) != digest:
            dest.unlink()
            raise SystemExit(f"Checksum mismatch for {name}; the partial file was removed - run again.")
        print(f"  ok           {dest.relative_to(ROOT)}")


def fetch_delay(raw_dir: Path) -> None:
    for t in DELAY_TRAINS:
        dest = raw_dir / f"{t}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ok (cached)  {dest.relative_to(ROOT)}")
            continue
        download(f"{HF}/{DELAY_REPO}/resolve/main/trains/{t}.csv", dest)
        print(f"  ok           {dest.relative_to(ROOT)}")


# ------------------------------------------------------------------ images
def stratified_split(labels: pd.Series, seed: int = SEED) -> pd.Series:
    """70/15/15 split per class, deterministic."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=labels.index)
    for _, idx in labels.groupby(labels).groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n_val = round(len(idx) * SPLIT_FRACTIONS[1])
        n_test = round(len(idx) * SPLIT_FRACTIONS[2])
        split[idx[:n_test]] = "test"
        split[idx[n_test:n_test + n_val]] = "validation"
    return split


def extract_images(raw_dir: Path, image_dir: Path, out_csv: Path) -> pd.DataFrame:
    rows, seen, dropped = [], set(), {"undecodable": 0, "duplicate_image": 0, "unknown_language": 0}
    for name in SIGN_FILES:
        src_split = name.split("-")[0]
        df = pd.read_parquet(raw_dir / name)
        missing = {"image", "language", "transcription"} - set(df.columns)
        if missing:
            raise SystemExit(f"{name}: expected columns missing: {sorted(missing)}")
        for rec in df.itertuples(index=False):
            lang = str(rec.language).strip().lower()
            if lang not in LANGUAGE_TO_SCRIPT:
                dropped["unknown_language"] += 1
                continue
            data = rec.image["bytes"] if isinstance(rec.image, dict) else None
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
                fmt = (img.format or "PNG").lower()
            except Exception:
                dropped["undecodable"] += 1
                continue
            digest = hashlib.sha1(data).hexdigest()
            if digest in seen:            # identical file seen before: keep one copy only
                dropped["duplicate_image"] += 1
                continue
            seen.add(digest)
            script = LANGUAGE_TO_SCRIPT[lang]
            ext = "jpg" if fmt in ("jpeg", "jpg") else fmt
            path = image_dir / script / f"{script.lower()}_{digest[:12]}.{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)       # original bytes, no re-encoding
            rows.append({"image_path": path.relative_to(ROOT).as_posix(), "script_label": script,
                         "language_label": lang.capitalize(), "transcription": rec.transcription,
                         "width": img.width, "height": img.height, "sha1": digest,
                         "source_split": src_split})
    meta = pd.DataFrame(rows)
    # Split on the deduplicated pool, stratified by LANGUAGE (so both the 3-script and
    # the 4-language experiments see every language in every split).
    meta["split"] = stratified_split(meta["language_label"])
    meta["source_dataset"] = SIGN_REPO
    meta["source_url"] = f"{HF}/{SIGN_REPO}"
    meta["license"] = "CC BY 4.0 (attribution required; cite the INDICSTR12 paper)"
    meta = meta[["image_path", "script_label", "language_label", "split", "transcription", "width",
                 "height", "sha1", "source_split", "source_dataset", "source_url", "license"]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"  images kept: {len(meta)}; dropped: {dropped}")
    print(meta.groupby(["language_label", "split"]).size().unstack(fill_value=0).to_string())
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Download RailScan raw datasets.")
    ap.add_argument("--skip-signboards", action="store_true")
    ap.add_argument("--skip-delay", action="store_true")
    args = ap.parse_args(argv)
    if not args.skip_signboards:
        print("Signboard images:")
        fetch_signboards(ROOT / "data" / "raw" / "signboards")
        extract_images(ROOT / "data" / "raw" / "signboards", ROOT / "data" / "sign_images",
                       ROOT / "data" / "processed" / "RailScan_CNN_Metadata.csv")
    if not args.skip_delay:
        print("Railway running records:")
        fetch_delay(ROOT / "data" / "raw" / "delay")


if __name__ == "__main__":
    sys.exit(main())
