# 0001 Streamlit modular monolith

**Context.** One developer, a few users at a time, a machine-learning core (CNN, OCR, delay models) in Python.

**Decision.** A single Streamlit app (`app.py`) for the UI, calling plain Python modules in `src/` that hold
all logic and are tested without Streamlit.

**Alternatives.** FastAPI back end + React front end: a real API and more UI control, but two codebases and
more hosting; not needed at this size.

**Consequences.** Fast to build and host for free. No public API. Long work (a scan) runs inside the user's
request; Streamlit reruns the script on each interaction, so slow steps must be cached.
