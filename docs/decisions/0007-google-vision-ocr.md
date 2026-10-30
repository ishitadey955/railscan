# 0007 Google Cloud Vision OCR, with EasyOCR as fallback

**Context.** EasyOCR's Devanagari reader cannot read some common letter combinations in sign fonts (on the
owner's Ratnagiri photo it never produced "रत्नागिरी", whatever the preprocessing: enlarging, contrast,
straightening, detector settings; 30 Oct 2026). Retraining it needs a GPU, thousands of labelled words and
days of work. The owner chose Google Cloud Vision.

**Decision.** When `GOOGLE_VISION_API_KEY` is set, the photo is read by Google Cloud Vision
(`src/ocr/google_vision.py`, TEXT_DETECTION with hints hi, mr, bn, kn, te, en); its text is shown and used for
station and train-number matching, with EasyOCR's readings as extra evidence. The script decision stays the
validated CNN + EasyOCR fusion (decision unchanged until Google is measured on it). No key, a failure, or more
than 1,000 photos in 30 days (Google's free tier; `users.RATE_LIMITS["google_ocr"]`) -> EasyOCR, silently.

**Evidence.** Request format and error handling checked against the real API (invalid key -> clean
"API key not valid", key never in messages); unit tests with recorded answers. Accuracy against EasyOCR on the
197 held-out photos: `python scripts/evaluate_google_ocr.py` (needs the key) -> reports/google_ocr_evaluation.json.
To be run once the key exists; until then no accuracy claim is made for Google.

**Consequences.** Scanned photos are sent to Google (privacy note in the app says so). Needs a Google Cloud
project with billing enabled; cost stays at zero within the free tier because of the cap. No new Python
dependency (standard-library HTTP).
