# Threat model

## Assets
User accounts (password hashes), users' history, the accounts database credentials, the host's CPU
allowance (a burst of scans can get the app throttled for everyone).

## Entry points and threats (OWASP Top 10)
| Entry point | Threat | Mitigation in place | Gap |
|---|---|---|---|
| Log-in form | Password guessing (A07) | scrypt, generic error, lockout 5 per 15 min per username, equal timing for unknown users | No per-IP limit on log-in; no 2FA |
| Sign-up form | Spam accounts (A07) | 5 sign-ups per hour per IP, username/password rules | No CAPTCHA |
| History (view/delete/export) | Reading or deleting another user's data (A01) | every query filters by the signed-in user's id; ids never come from the URL | — |
| Photo upload | Huge or malformed files (A04) | 10 MB server limit; images over 60 megapixels refused from the file header before decoding; decompression bombs and decoding errors become a clean message; photos never stored | — |
| Scan button | CPU exhaustion | 10 scans per 10 min per user, in the database | — |
| Station search / OCR text | Injection (A03), XSS | parameterised SQL, OCR text shown with `st.code`/markdown (no raw HTML) | — |
| Gemini OCR (when enabled) | Photos leave RailScan and are used by Google on the free tier; invented text | key only in Secrets/header; privacy note; stations from Gemini text always need the user's confirmation | Google's data use on the free tier |
| Google OCR (when enabled) | Photos leave RailScan; key leak; cost | key only in Secrets and in a request header (never in URLs or logs); 1,000 photos per 30 days cap, then EasyOCR; privacy note tells users | Google's own data handling (their terms) |
| Configuration | Leaked secrets (A02, A05) | secrets only in Streamlit Secrets / env; examples committed; gitleaks in CI; history scanned clean (30 Oct 2026) | — |
| Errors | Information leakage (A05) | friendly messages; details only in redacted logs; `showErrorDetails = "type"` | — |
| Dependencies | Vulnerable packages (A06) | pinned versions | No automated audit yet (pip-audit / Dependabot) |
| Sessions | Hijacking, CSRF | Streamlit websocket session, XSRF protection on | No server-side session revocation (log-out clears the session only in that browser) |

## Not covered
No external penetration test. Before handling sensitive personal data at scale, get one and revisit
decision 0005 (auth provider).
