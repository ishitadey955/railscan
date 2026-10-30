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


def test_file_cache_does_not_count_as_used(monkeypatch, tmp_path):
    # what Streamlit Cloud reported after start-up: 2,919 of 3,072 MB "used", mostly the file cache
    # of the freshly installed packages (active and inactive), which the kernel drops when needed
    _cgroup(monkeypatch, tmp_path, str(3072 * res.MB), 2919 * res.MB)
    (tmp_path / "memory.stat").write_text(
        f"anon {250 * res.MB}\nactive_file {1900 * res.MB}\ninactive_file {700 * res.MB}\nshmem 0\n")
    assert res.memory_used_mb() == 319
    assert res.has_room_for(1500)                 # 319 + 1500 <= 0.9 x 3072; raw usage refused every scan


def test_cgroup_v1_file_cache_is_subtracted_too(monkeypatch, tmp_path):
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.limit_in_bytes").write_text(str(3072 * res.MB))
    (v1 / "memory.usage_in_bytes").write_text(str(2000 * res.MB))
    (v1 / "memory.stat").write_text(f"total_active_file {800 * res.MB}\ntotal_inactive_file {700 * res.MB}\n")
    monkeypatch.setattr(res, "_V2", tmp_path / "no-v2")
    monkeypatch.setattr(res, "_V1", v1)
    assert res.memory_limit_mb() == 3072 and res.memory_used_mb() == 500
