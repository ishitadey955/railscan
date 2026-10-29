#!/usr/bin/env python3
"""
RailScan - add Bengali signboard photos for the script CNN.

Source : darknight054/indicstr12-full on Hugging Face, subset `bengali` (the full-image part
         of the real IndicSTR12 dataset, CVIT IIIT Hyderabad; Lunia, Mondal, Jawahar, ICDAR 2023).
         Pinned to revision REVISION; every file is checked against its SHA-256.
License: the original project page (https://cvit.iiit.ac.in/research/projects/cvit-projects/indicstr)
         states no licence; the Hugging Face copy refers to it. The other RailScan images come
         from a copy of the same dataset that lists CC BY 4.0. Used here for research with
         attribution (cite the IndicSTR12 paper).

What it does:
  * downloads the three Bengali parquet files to data/raw/signboards_bengali/
  * keeps each distinct image once (SHA-1 of the original bytes), saves the ORIGINAL bytes to
    data/sign_images/Bengali/
  * transcription = the word labels joined with spaces
  * splits the Bengali pool 70/15/15 with the same rule and seed as the other scripts
  * appends the rows to data/processed/RailScan_CNN_Metadata.csv; existing rows are not changed

Usage:  python scripts/add_bengali_signboards.py
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_datasets as dd  # noqa: E402

ROOT = dd.ROOT
REPO = "darknight054/indicstr12-full"
REVISION = "b3c56200ac503d3b625204ca8c2dd6ebc4d07e09"
FILES = {  # file -> Git-LFS SHA-256 from the Hugging Face file listing
    "test-00000-of-00001.parquet": "cb366edfb715a47c593bb2d45bbbc54a1415b3317900633bafa08a9e2cbe2ce3",
    "train-00000-of-00001.parquet": "3138dc018524a48a7962c14d76a4caaa0ae81f709c8e1bea7eca63fa5d73fbd3",
    "validation-00000-of-00001.parquet": "12838337531282fb432b96ad87776505a4ee24e038545bf2b5cf3973e62233f4",
}
RAW = ROOT / "data" / "raw" / "signboards_bengali"
IMAGES = ROOT / "data" / "sign_images"
META = ROOT / "data" / "processed" / "RailScan_CNN_Metadata.csv"
LICENSE = ("not stated by the original source (CVIT IndicSTR12); used for research with attribution - "
           "cite the IndicSTR12 paper")


def fetch() -> None:
    for name, digest in FILES.items():
        dest = RAW / name
        if dest.exists() and dd.sha256(dest) == digest:
            print(f"  ok (cached)  {dest.relative_to(ROOT)}")
            continue
        print(f"  downloading  bengali/{name}")
        dd.download(f"{dd.HF}/{REPO}/resolve/{REVISION}/bengali/{name}", dest)
        if dd.sha256(dest) != digest:
            dest.unlink()
            raise SystemExit(f"Checksum mismatch for {name}; the partial file was removed - run again.")


def main() -> None:
    fetch()
    meta = pd.read_csv(META)
    meta = meta[meta.script_label != "Bengali"]          # re-running replaces only the Bengali rows
    seen = set(meta.sha1)
    rows, dropped = [], {"undecodable": 0, "duplicate_image": 0, "no_text": 0}
    for name in FILES:
        df = pd.read_parquet(RAW / name)
        for rec in df.itertuples(index=False):
            data = rec.image["bytes"] if isinstance(rec.image, dict) else None
            words = [str(w).strip() for w in (rec.text if rec.text is not None else []) if str(w).strip()]
            if not words:
                dropped["no_text"] += 1
                continue
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
                fmt = (img.format or "PNG").lower()
            except Exception:
                dropped["undecodable"] += 1
                continue
            digest = hashlib.sha1(data).hexdigest()
            if digest in seen:
                dropped["duplicate_image"] += 1
                continue
            seen.add(digest)
            ext = "jpg" if fmt in ("jpeg", "jpg") else fmt
            path = IMAGES / "Bengali" / f"bengali_{digest[:12]}.{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)                   # original bytes, no re-encoding
            rows.append({"image_path": path.relative_to(ROOT).as_posix(), "script_label": "Bengali",
                         "language_label": "Bengali", "transcription": " ".join(words),
                         "width": img.width, "height": img.height, "sha1": digest,
                         "source_split": name.split("-")[0], "source_dataset": f"{REPO}@{REVISION[:7]}",
                         "source_url": f"{dd.HF}/{REPO}", "license": LICENSE})
    new = pd.DataFrame(rows)
    new["split"] = dd.stratified_split(new["language_label"]).values
    out = pd.concat([meta, new[meta.columns]], ignore_index=True)
    out.to_csv(META, index=False, encoding="utf-8")
    print(f"  Bengali images added: {len(new)}; dropped: {dropped}")
    print(out.groupby(["script_label", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
