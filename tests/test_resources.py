"""Memory headroom checks read the container's cgroup files (faked here with a temporary folder)."""
import src.resources as res


def _cgroup(monkeypatch, tmp_path, limit, current):
    (tmp_path / "memory.max").write_text(limit)
    (tmp_path / "memory.current").write_text(str(current))
    monkeypatch.setattr(res, "_V2", tmp_path)
    monkeypatch.setattr(res, "_V1", tmp_path / "missing")


def test_limit_and_use_are_read_from_cgroup_v2(monkeypatch, tmp_path):
    _cgroup(monkeypatch, tmp_path, str(2700 * res.MB), 1000 * res.MB)
    assert res.memory_limit_mb() == 2700 and res.memory_used_mb() == 1000


def test_room_is_refused_near_the_limit(monkeypatch, tmp_path):
    _cgroup(monkeypatch, tmp_path, str(2000 * res.MB), 1500 * res.MB)
    assert res.has_room_for(200)                  # 1700 <= 0.9 x 2000
    assert not res.has_room_for(400)              # 1900 >  1800


def test_no_limit_means_nothing_is_refused(monkeypatch, tmp_path):
    _cgroup(monkeypatch, tmp_path, "max", 5000 * res.MB)
    assert res.memory_limit_mb() is None and res.has_room_for(10_000)
