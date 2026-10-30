"""Unloading the scanner models must give the memory back on the server's OS (Linux / glibc).

The app loads the models in a background thread and unloads them after a quiet period
(app.py: idle_unloader). On Windows the objects are freed but the C runtime keeps the pages for
reuse, so this is only measured on Linux, where it matters (it runs in CI). It downloads the
EasyOCR models on first use.
"""
import gc
import sys
import threading

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="measures the server's (Linux/glibc) behaviour")


def _rss_mb() -> float:
    with open("/proc/self/status") as fh:
        return next(int(line.split()[1]) for line in fh if line.startswith("VmRSS:")) / 1024


def test_unloaded_models_return_their_memory():
    import ctypes

    pytest.importorskip("easyocr")
    from src.image.prediction import load_model, unload_model
    from src.ocr import language_candidates
    from src.ocr.ocr_engine import preload_readers, unload_readers
    from src.script_evidence import SCRIPTS

    libc = ctypes.CDLL("libc.so.6")
    libc.mallopt(-8, 2)                              # M_ARENA_MAX, as the app sets it
    before = _rss_mb()
    t = threading.Thread(target=lambda: (load_model(), preload_readers(
        [language_candidates(s)[0] for s in SCRIPTS])))
    t.start()
    t.join()
    loaded = _rss_mb()
    unload_readers()
    unload_model()
    gc.collect()
    libc.malloc_trim(0)
    after = _rss_mb()
    print(f"RSS before {before:.0f} MB, loaded {loaded:.0f} MB, after unload {after:.0f} MB")
    assert loaded - before > 300, "models did not load"
    assert loaded - after >= 0.5 * (loaded - before), "less than half of the models' memory came back"
