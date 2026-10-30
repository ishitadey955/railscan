# 0008 Gemini OCR (free API key) as an alternative to Cloud Vision

**Context.** Cloud Vision (0007) needs a Google Cloud billing account; the owner prefers not to link one.
A Gemini API key from Google AI Studio works without billing on the free tier (31 Oct 2026).

**Decision.** `src/ocr/gemini_ocr.py`: the photo goes to the Gemini API (`gemini-3.6-flash` by default,
`GEMINI_OCR_MODEL` to change) with an instruction to copy the sign's text exactly, line by line, in its
original scripts, without guessing, translating or watermarks; JSON output, temperature 0. Order in the app:
Cloud Vision if its key is set, then Gemini, then EasyOCR. The page says why Google was not used.
A station found from Gemini's text is never accepted automatically: a language model can produce the name it
expects rather than what is printed, so the user confirms it.

**Consequences.** Free-tier content is used by Google to improve its products (Gemini API pricing page); the
privacy note says so. Free-tier rate limits apply (HTTP 429 -> EasyOCR). No confidence score from Gemini.
Accuracy, including invented words (word precision), is measured with
`python scripts/evaluate_google_ocr.py --engine gemini` once the key exists; no claim before that.
