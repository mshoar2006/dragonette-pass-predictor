"""Tests for the silent-wrong-answer and operational-risk fixes of 2026-07-15.

Each test here corresponds to a defect that produced a *confidently wrong* result
or an unhandled traceback rather than an error. Fully offline. [SESSION 2026-07-15]
"""
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIX = ROOT / "fixtures"
sys.path.insert(0, str(SRC))
import passes as P  # noqa: E402

SITEA_KMZ = (FIX / "SiteA.kmz").read_bytes()
DEMO_TLES = str(FIX / "demo_tles_synthetic.txt")
START = datetime(2026, 7, 14, 0, 0, tzinfo=timezone.utc)


def _tles():
    return P._parse_3le_file(Path(DEMO_TLES).read_text(), P.SATELLITES)


# ------------------------------------------------------- --start / window parsing
def test_parse_start_utc_converts_an_offset_instead_of_relabelling_it():
    """The bug: `fromisoformat(s).replace(tzinfo=utc)` parsed +10:00 then threw it
    away, silently shifting a Brisbane user's window by 10 h."""
    got = P.parse_start_utc("2026-08-01T00:00:00+10:00")
    assert got == datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)
    # naive input is assumed UTC
    assert P.parse_start_utc("2026-08-01T00:00:00") == datetime(
        2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    # negative offsets too
    assert P.parse_start_utc("2026-08-01T00:00:00-05:00") == datetime(
        2026, 8, 1, 5, 0, tzinfo=timezone.utc)
    assert P.parse_start_utc(None) is None and P.parse_start_utc("") is None


def test_parse_start_utc_rejects_garbage():
    with pytest.raises(ValueError):
        P.parse_start_utc("not-a-date")


def test_validate_window_bounds_match_the_documented_api_limit():
    P.validate_window(14.0)
    P.validate_window(P.MAX_WINDOW_DAYS)
    for bad in (0.0, -5.0, P.MAX_WINDOW_DAYS + 0.1, 10000.0):
        with pytest.raises(ValueError):
            P.validate_window(bad)


def test_start_offset_actually_shifts_the_predicted_window():
    """End-to-end: the offset must reach predict(), not just parse correctly."""
    kw = dict(days=1.0, terrain_alt_m=400.0, polygon_name="SITEA_100sqkm",
              tles=_tles())
    utc = P.predict(SITEA_KMZ, start_utc=P.parse_start_utc("2026-07-14T00:00:00"), **kw)
    off = P.predict(SITEA_KMZ, start_utc=P.parse_start_utc("2026-07-14T00:00:00+10:00"), **kw)
    assert (utc.start_utc - off.start_utc) == timedelta(hours=10)


def test_cli_start_with_offset_is_honoured(tmp_path):
    """cli.py --start used to reinterpret an offset as UTC."""
    out = tmp_path / "o.xlsx"
    r = subprocess.run(
        [sys.executable, str(SRC / "cli.py"), str(FIX / "SiteA.kmz"),
         "--polygon", "SITEA_100sqkm", "--alt", "400", "--days", "1",
         "--tle-file", DEMO_TLES, "--start", "2026-07-14T10:00:00+10:00",
         "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "2026-07-14 00:00Z" in r.stdout, f"expected the +10:00 start converted to 00:00Z\n{r.stdout}"


# --------------------------------------------------------------- CLI clean errors
@pytest.mark.parametrize("args,expect", [
    (["--start", "garbage"], "error:"),
    (["--days", "0"], "days must be"),
    (["--days", "9999"], "days must be"),
    (["--tz", "Mars/Olympus"], "unknown --tz"),
    (["--cloud-threshold", "500"], "cloud-threshold"),
])
def test_cli_reports_bad_input_without_a_traceback(args, expect, tmp_path):
    r = subprocess.run(
        [sys.executable, str(SRC / "cli.py"), str(FIX / "SiteA.kmz"),
         "--polygon", "SITEA_100sqkm", "--tle-file", DEMO_TLES,
         "-o", str(tmp_path / "x.xlsx")] + args,
        capture_output=True, text=True)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    assert expect.lower() in r.stderr.lower()


def test_cli_missing_file_is_a_clean_error(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SRC / "cli.py"), str(tmp_path / "nope.kmz"),
         "--tle-file", DEMO_TLES], capture_output=True, text=True)
    assert r.returncode == 2 and "Traceback" not in r.stderr
    assert "no such file" in r.stderr.lower()


# --------------------------------------------------- API honours the Start field
def test_api_accepts_start_and_uses_it(monkeypatch):
    """index.html posts `start`, but no endpoint declared it and FastAPI silently
    drops unknown form fields — so the UI's "Start (UTC)" box was a no-op: a
    planner set a future campaign date and got a window starting *now*, with no
    error shown. [SESSION 2026-07-15]"""
    from fastapi.testclient import TestClient
    import app as A
    monkeypatch.setattr(P, "fetch_tles", lambda *a, **k: (_tles(), []))
    client = TestClient(A.app)
    r = client.post("/predict/json",
                    files={"kmz": ("t.kmz", SITEA_KMZ)},
                    data={"days": "1", "alt": "400", "min_sun": "-90",
                          "polygon": "SITEA_100sqkm",
                          "start": "2026-07-14T00:00:00"})
    assert r.status_code == 200, r.text
    assert r.json()["window_utc"][0].startswith("2026-07-14T00:00")


def test_api_start_offset_is_converted_not_relabelled(monkeypatch):
    from fastapi.testclient import TestClient
    import app as A
    monkeypatch.setattr(P, "fetch_tles", lambda *a, **k: (_tles(), []))
    client = TestClient(A.app)
    r = client.post("/predict/json",
                    files={"kmz": ("t.kmz", SITEA_KMZ)},
                    data={"days": "1", "alt": "400", "min_sun": "-90",
                          "polygon": "SITEA_100sqkm",
                          "start": "2026-07-14T10:00:00+10:00"})
    assert r.status_code == 200, r.text
    assert r.json()["window_utc"][0].startswith("2026-07-14T00:00")


@pytest.mark.parametrize("data,code", [
    ({"start": "garbage"}, 422),
    ({"days": "0"}, 422),
    ({"days": "99"}, 422),
    ({"tz": "Mars/Olympus"}, 422),
])
def test_api_rejects_bad_input_with_422(data, code, monkeypatch):
    """`tz` used to be validated only after the full prediction (and reported as
    a 422 that conflated user error with an internal failure)."""
    from fastapi.testclient import TestClient
    import app as A
    monkeypatch.setattr(P, "fetch_tles", lambda *a, **k: (_tles(), []))
    base = {"days": "1", "alt": "400", "min_sun": "-90",
            "polygon": "SITEA_100sqkm"}
    r = TestClient(A.app).post("/predict/json",
                               files={"kmz": ("t.kmz", SITEA_KMZ)},
                               data={**base, **data})
    assert r.status_code == code, r.text


# ------------------------------------------------------------ SGP4 error surfacing
def test_sgp4_errors_are_reported_not_swallowed():
    """A decayed element set gave 0 passes and 0 warnings — byte-identical to a
    satellite that genuinely had no opportunities."""
    tles = _tles()
    name = sorted(tles)[0]
    good = tles[name]
    # bstar wildly high + high mean motion => SGP4 raises error codes when propagated
    l1 = good.line1[:53] + " 99999-0" + good.line1[61:]
    decayed = P.TLE(good.name, good.catnr, l1, good.line2, good.fetched_utc)
    pred = P.predict(SITEA_KMZ, days=3.0, start_utc=START, terrain_alt_m=400.0,
                     polygon_name="SITEA_100sqkm", tles={name: decayed})
    joined = " ".join(pred.warnings)
    assert "SGP4" in joined, f"a propagation failure must be stated; got {pred.warnings}"


def test_empty_tle_set_returns_an_empty_prediction_not_a_crash():
    """Used to raise `max() iterable argument is empty`."""
    pred = P.predict(SITEA_KMZ, days=1.0, start_utc=START, terrain_alt_m=400.0,
                     polygon_name="SITEA_100sqkm", tles={})
    assert pred.passes == [] and pred.marginal == [] and pred.nonoperational == []
    assert any("No TLEs available" in w for w in pred.warnings)
    assert P.write_xlsx_multi([pred], "Australia/Brisbane")      # outputs still work
    assert P.prediction_json([pred])["passes"] == []


# ------------------------------------------------------------ manoeuvre detection
# The fixture PAIR is the point: DRAG04 manoeuvred between these two real element
# sets (semi-major +113 m over ~1.1 d while every sibling decayed 6-18 m), so they
# pin the detector against an actual burn rather than a synthetic one.
# [VERIFIED 2026-07-15 — see IMPROVEMENTS.md A4-bis.]
OLD_TLES = str(FIX / "tles_real_20260714.txt")
NEW_TLES = str(FIX / "tles_real_20260715.txt")


def _pair():
    return (P._parse_3le_file(Path(OLD_TLES).read_text(), P.SATELLITES),
            P._parse_3le_file(Path(NEW_TLES).read_text(), P.SATELLITES))


def test_orbit_change_flags_the_real_drag04_burn():
    old, new = _pair()
    ch = P.orbit_change(old["DRAG04"], new["DRAG04"])
    assert ch["manoeuvred"] is True
    assert ch["da_km"] > 0.05, "DRAG04 raised its orbit; drag cannot do that"
    # the actionable consequence: the superseded set was ~20 km / ~2.6 s out
    assert ch["pos_err_km"] > 10.0
    assert ch["along_track_s"] > 1.0


def test_orbit_change_does_not_flag_natural_decay():
    """The four non-manoeuvring satellites must stay silent, or the warning is
    noise and gets ignored."""
    old, new = _pair()
    for name in ("DRAG01", "DRAG02", "DRAG03", "DRAG05"):
        ch = P.orbit_change(old[name], new[name])
        assert ch["manoeuvred"] is False, f"{name} false-positived: {ch}"
        assert ch["da_km"] < 0, f"{name} should be decaying, not rising: {ch}"
        assert ch["pos_err_km"] < 2.0, f"{name} free-flight error: {ch}"


def test_orbit_change_rejects_unusable_pairs():
    old, new = _pair()
    assert P.orbit_change(new["DRAG01"], old["DRAG01"]) is None    # reversed
    assert P.orbit_change(old["DRAG01"], old["DRAG01"]) is None    # same epoch


def test_fetch_tles_warns_when_a_satellite_has_manoeuvred(tmp_path):
    """End-to-end: the cache is the only orbit history the tool has, so the
    comparison must happen at fetch time, before the old set is overwritten."""
    old, new = _pair()
    cache = tmp_path / "tles.json"
    P._save_cache(cache, old)
    blob = P._load_cache(cache)
    blob["_ts"] = time.time() - 99 * 3600          # force stale -> triggers a fetch
    P._write_cache_atomic(cache, blob)

    def fake_get(url):
        catnr = int(url.split("CATNR=")[1].split("&")[0])
        t = next(x for x in new.values() if x.catnr == catnr)
        return f"{t.name}\n{t.line1}\n{t.line2}\n"

    tles, warns = P.fetch_tles(satellites=P.SATELLITES, cache_path=cache,
                               http_get=fake_get)
    assert set(tles) == set(P.SATELLITES)
    man = [w for w in warns if "MANOEUVRE" in w]
    assert len(man) == 1, f"expected exactly one manoeuvre warning, got {warns}"
    assert "DRAG04" in man[0]
    assert "113 m" in man[0] and "19.9 km" in man[0]
    # and it must say why the age-based sigma cannot be trusted for this satellite
    assert "σ" in man[0] or "sigma" in man[0].lower()


def test_no_manoeuvre_warning_without_history(tmp_path):
    """A cold cache has nothing to compare against — it must stay quiet rather
    than guess."""
    _, new = _pair()

    def fake_get(url):
        catnr = int(url.split("CATNR=")[1].split("&")[0])
        t = next(x for x in new.values() if x.catnr == catnr)
        return f"{t.name}\n{t.line1}\n{t.line2}\n"

    _, warns = P.fetch_tles(satellites=P.SATELLITES,
                            cache_path=tmp_path / "cold.json", http_get=fake_get)
    assert not [w for w in warns if "MANOEUVRE" in w]


# ------------------------------------------------- Celestrak politeness / stampede
def test_celestrak_fetch_sends_a_user_agent():
    """Celestrak blocks generic scripted agents; a 403 would silently degrade
    every prediction to stale elements."""
    assert "User-Agent" in P.CELESTRAK_UA
    assert "dragonette" in P.CELESTRAK_UA["User-Agent"].lower()


def test_failed_fetch_is_negatively_cached_so_an_outage_is_not_amplified(tmp_path):
    """Without this, an outage turns each request into 5 more — the retry storm
    that earns an IP ban."""
    cache = tmp_path / "tles.json"
    P._save_cache(cache, _tles())                 # a stale-but-usable cache exists
    blob = P._load_cache(cache)
    blob["_ts"] = time.time() - 99 * 3600         # force it stale
    P._write_cache_atomic(cache, blob)

    calls = []

    def failing_get(url):
        calls.append(url)
        raise RuntimeError("503 Service Unavailable")

    t1, w1 = P.fetch_tles(cache_path=cache, http_get=failing_get)
    assert t1 and any("Celestrak fetch failed" in w for w in w1)
    n_after_first = len(calls)
    assert n_after_first > 0, "the first attempt should really try"

    # Second call within the cooldown must NOT re-hit the network.
    t2, w2 = P.fetch_tles(cache_path=cache, http_get=failing_get)
    assert t2, "stale cache should still be served"
    assert len(calls) == n_after_first, "cooldown must suppress the retry storm"
    assert any("not retrying" in w for w in w2)


def test_fetch_failure_record_preserves_the_cached_tles(tmp_path):
    cache = tmp_path / "tles.json"
    P._save_cache(cache, _tles())
    P._record_fetch_failure(cache)
    blob = P._load_cache(cache)
    assert blob["_fail_ts"] > 0
    assert P._cache_to_tles(blob, P.SATELLITES), "TLEs must survive the failure stamp"


def test_cache_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    cache = tmp_path / "tles.json"
    P._save_cache(cache, _tles())
    assert cache.is_file()
    assert not list(tmp_path.glob("*.tmp")), "temp file must be replaced, not left behind"
    assert P._load_cache(cache) is not None
