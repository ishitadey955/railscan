"""How much memory this server allows and uses, so the app can refuse work instead of being killed.

Streamlit Community Cloud stops an app that goes over its memory limit (the page then shows
"Oh no. Error running app"). The limit is the container's cgroup limit; this module reads it and
the container's current use from Linux's cgroup files (v2, else v1). Elsewhere (Windows, macOS,
no limit set) the limit is unknown and nothing is refused.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

MB = 1024 * 1024
_V2 = Path("/sys/fs/cgroup")
_V1 = Path("/sys/fs/cgroup/memory")


def _read_int(path: Path) -> Optional[int]:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None          # "max" (v2) = no limit


def memory_limit_mb() -> Optional[float]:
    """The container's memory limit in MB, or None when there is none / it is not readable."""
    for path in (_V2 / "memory.max", _V1 / "memory.limit_in_bytes"):
        value = _read_int(path)
        if value is not None and value < 1 << 50:           # v1 reports a huge number for "no limit"
            return value / MB
    return None


def memory_used_mb() -> Optional[float]:
    """Memory the container uses now (what the limit is checked against), else this process's RSS."""
    for path in (_V2 / "memory.current", _V1 / "memory.usage_in_bytes"):
        value = _read_int(path)
        if value is not None:
            return value / MB
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / MB
    except Exception:
        return None


def has_room_for(need_mb: float, safety: float = 0.9) -> bool:
    """True unless a known limit would be passed: used + need must stay below ``safety`` x limit."""
    limit, used = memory_limit_mb(), memory_used_mb()
    if limit is None or used is None:
        return True
    return used + need_mb <= safety * limit
