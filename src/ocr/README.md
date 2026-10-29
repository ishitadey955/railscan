# RailScan OCR component

Reads the text on a sign image with **EasyOCR** (PyTorch, CPU). It accepts a file path, uploaded bytes, a PIL image or a NumPy array. Measured accuracy on held-out signboards is in `reports/ocr_evaluation.json` (main README, section 7).

```
sign image ─► CNN script detection ─► script→language(s) ─► preprocessing variants
          ─► EasyOCR (per language × variant, best kept) ─► text cleaning
          ─► train-number split ─► result dict
```

## 1. Install
EasyOCR is part of the main `requirements.txt`. For the OCR component on its own:
```bash
pip install -r requirements-ocr.txt      # easyocr, CPU-only torch/torchvision, opencv, numpy, Pillow, pytest
```
Both files point pip at PyTorch's CPU index, so neither Windows nor Linux downloads the multi-GB CUDA build.

There is no separate program to install. On first use, EasyOCR downloads its models to `~/.EasyOCR/model`: one text detector, plus one recognition model each for Devanagari, Kannada, Telugu and English (roughly 100 MB per script). After that it works offline.

Verify the setup:
```bash
python scripts/test_ocr.py --check
python -m pytest tests/test_ocr.py -v -rs
```

## 2. Use
```python
from src.ocr import extract_text, OCREngine, OCRConfig

r = extract_text("sign.jpg")                      # CNN picks the language
r = extract_text("sign.jpg", script="Hindi")      # skip the CNN
r = extract_text("sign.jpg", languages="eng+kan") # force a language string

engine = OCREngine(OCRConfig(languages_mode="both", deskew=True))   # reuse for many images
r = engine.recognize("sign.jpg", return_attempts=True)
```
CLI: `python scripts/test_ocr.py sign.jpg [--script Hindi] [--lang eng+hin] [--attempts] [--json]`

The engine never raises for bad input. Failures come back as a normal result.

### Result
```json
{
  "status": "ok",
  "text": "नई दिल्ली",            // == clean_text ("" when OCR failed)
  "raw_text": "नई दिल्ली",        // exactly what EasyOCR returned, boxes joined into lines
  "clean_text": "नई दिल्ली",
  "ocr_language": "eng+hin+mar",
  "ocr_confidence": 43.5,          // mean EasyOCR box confidence 0-100; null if failed
  "ocr_backend": "easyocr",
  "preprocessing_method": "original",
  "low_confidence": true,          // confidence < OCRConfig.min_confidence (60)
  "train_numbers": ["12301"],      // five-digit candidates, ASCII digits
  "station_text": "नई दिल्ली",     // clean_text without train numbers → input for station matching
  "script": "Devanagari", "script_confidence": 0.42, "script_source": "cnn",
  "script_low_confidence": true,
  "warnings": [], "error": null, "image_analysis": {...}
}
```
This shows the format only; it is not a measured result.

**On failure:**
- `status` is `"OCR failed"`, `text` is `""` and `ocr_confidence` is `null`.
- `error` explains why: an unreadable file, EasyOCR not installed, only noise recognised, and so on.

**`script_source`** says where the script came from:
- `cnn`: detected by the CNN;
- `override`: you passed `script=`;
- `language_override`: you passed `languages=`;
- `unavailable`: the CNN could not run, so every supported script was tried and a warning was added.

## 3. How it works
**Language selection**
- Language codes are `eng, hin, mar, kan, tel`, mapped to EasyOCR's `en, hi, mr, kn, te`.
- The CNN class `Devanagari` maps to `hin+mar`.
- The default `languages_mode="bilingual"` adds English (for example `eng+kan`). `"native"` uses the script's language only; `"both"` tries both.
- EasyOCR can combine Hindi and Marathi with English in one reader. Kannada and Telugu need their own reader, so each language combination gets one cached reader.
- If the CNN is unsure, or the runner-up has a probability of 0.25 or more, the runner-up's languages are tried as well, and the better-scoring result wins.

**Preprocessing** (`preprocessing.py`)
- Available variants: original, grayscale, denoise, contrast (CLAHE), adaptive and Otsu threshold, 2× upscale, optional deskew, `upscale_otsu` and `inverted`.
- `analyze_image` decides the **order** in which variants are tried.
- EasyOCR takes seconds per call on CPU, so only the first `max_strategies` (default 2) variants are tried for each language.

**Choosing the result**
- Attempts stop early once one reaches `early_stop_confidence` (85).
- Otherwise the best `confidence × (share of letter/digit characters)` wins.
- Attempts below `min_accept_confidence` (30), or with fewer than 2 letters or digits, count as noise.
- These thresholds are heuristics, not measured values.

**Cleaning** (`text_cleaning.py`)
- NFC normalisation, removal of invisible characters and noise symbols, and duplicate-line removal.
- All Indic letters, matras, virama and native digits are kept.
- ZWJ/ZWNJ are removed from `clean_text` but kept in `raw_text`.

**Train numbers** (`train_numbers.py`)
- The regex `(?<!\d)\d{5}(?!\d)` runs after native digits are converted to ASCII.
- Matches are candidates only. The app accepts a number only if it is a train in the dataset.

## 4. Limits
- **Accuracy on scene photos is modest.** See `reports/ocr_evaluation.json`: about 29% of ground-truth words found, and no usable text on 36% of images. Clear station boards read much better.
- **Speed.** About 4 s per image on CPU. The first use of each script also downloads its model.

## 5. Tests
Run `python -m pytest tests/test_ocr.py -v -rs`.
- Control-flow tests replace the EasyOCR call with a stub, so they always run.
- Real-OCR tests draw known text with a system font and run real EasyOCR.
- The Indic real-OCR tests need Pillow with **libraqm** to shape Devanagari, Kannada and Telugu text. Without it they skip with a reason.
- If your system lacks suitable fonts, set `RAILSCAN_FONT_DIR` to a folder holding Noto Sans Devanagari, Kannada and Telugu fonts. Windows' Nirmala UI also works.
