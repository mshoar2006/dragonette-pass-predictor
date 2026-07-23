"""Offline tests for the cloud-forecast hindcast harness (src/hindcast.py).

Fully offline: real captured STAC + Open-Meteo responses are injected via the
`search_json`/`hindcast_json` seams, mirroring the `http_get` pattern in
passes.py. No network.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import hindcast as H  # noqa: E402

FIX = ROOT / "fixtures"
SCENES_JSON = json.loads(
    (FIX / "stac_scenes_siteA_2026-05-01_2026-06-30.json").read_text())
HINDCAST_JSON = json.loads(
    (FIX / "openmeteo_hindcast_siteA_2026-05-01_2026-06-30.json").read_text())
SITEA = (-20.0, 150.0)
START = datetime(2026, 5, 1, tzinfo=timezone.utc)
END = datetime(2026, 6, 30, tzinfo=timezone.utc)


def test_fetch_scenes_parses_real_stac_payload():
    scenes = H.fetch_scenes(*SITEA, START, END, search_json=SCENES_JSON)
    assert len(scenes) == 20
    assert all(isinstance(s.acquired_utc, datetime) for s in scenes)
    assert all(s.acquired_utc.tzinfo is not None for s in scenes)
    assert all(0.0 <= s.observed_cloud_pct <= 100.0 for s in scenes)
    # both constellations present -> the proxy spans Landsat and Sentinel-2
    plats = {s.platform for s in scenes}
    assert any(p.startswith("landsat") for p in plats)
    assert any(p.startswith("sentinel-2") for p in plats)
    assert scenes == sorted(scenes, key=lambda s: s.acquired_utc)


def test_scenes_without_observed_cloud_are_dropped():
    """A scene with no eo:cloud_cover cannot be scored, so it must not appear
    as if it had 0% cloud."""
    payload = {"features": [
        {"collection": "c", "properties": {"datetime": "2026-05-03T00:04:23Z",
                                           "platform": "x", "eo:cloud_cover": None}},
        {"collection": "c", "properties": {"datetime": "2026-05-04T00:04:23Z",
                                           "platform": "x"}},
        {"collection": "c", "properties": {"datetime": "2026-05-05T00:04:23Z",
                                           "platform": "y", "eo:cloud_cover": 12.5}},
    ]}
    scenes = H.fetch_scenes(*SITEA, START, END, search_json=payload)
    assert [s.platform for s in scenes] == ["y"]
    assert scenes[0].observed_cloud_pct == 12.5


def test_lead_beyond_provider_cap_is_rejected():
    """Open-Meteo serves forecasts issued at most 7 days ahead; day 8+ returns an
    empty series. Asking for more must fail loudly, not silently score nothing —
    checked against the live API."""
    with pytest.raises(ValueError):
        H.fetch_hindcast(*SITEA, START, END, leads=(8,), hindcast_json=None)
    with pytest.raises(ValueError):
        H.fetch_hindcast(*SITEA, START, END, leads=(0,), hindcast_json=None)


def test_hindcast_url_requests_previous_day_fields_not_just_analysis():
    """The load-bearing subtlety: plain `cloud_cover` from the historical API is
    the analysis (measured 240/240 hours identical to ERA5), so scoring against
    it is circular. The URL must ask for the previous_dayN series."""
    url = H.hindcast_url(*SITEA, START, END, leads=(1, 3))
    assert "cloud_cover_previous_day1" in url
    assert "cloud_cover_previous_day3" in url
    assert "historical-forecast-api" in url


def test_pairing_joins_every_scene_at_every_lead():
    scenes = H.fetch_scenes(*SITEA, START, END, search_json=SCENES_JSON)
    pairings = H.pair(scenes, HINDCAST_JSON)
    assert len(pairings) == len(scenes) * H.MAX_LEAD_DAYS
    assert {p.lead_days for p in pairings} == set(H.LEADS)
    assert all(0.0 <= p.forecast_cloud_pct <= 100.0 for p in pairings)


def test_pairing_refuses_scenes_outside_the_series():
    """Same failure attach_cloud had: a scene past the end of the series must be
    skipped, never snapped to the last available hour."""
    scenes = H.fetch_scenes(*SITEA, START, END, search_json=SCENES_JSON)
    far = H.Scene(platform="sentinel-2c", collection="sentinel-2-l2a",
                  acquired_utc=scenes[-1].acquired_utc + timedelta(days=60),
                  observed_cloud_pct=10.0)
    only_far = H.pair([far], HINDCAST_JSON)
    assert only_far == [], "a scene outside the series must not be paired"
    # and it must not contaminate a batch containing valid scenes
    mixed = H.pair(scenes + [far], HINDCAST_JSON)
    assert len(mixed) == len(scenes) * H.MAX_LEAD_DAYS


def test_forecast_series_actually_differs_from_the_analysis():
    """Guards the whole method: if previous_dayN ever collapses onto the plain
    analysis, every skill number becomes meaningless-but-excellent."""
    h = HINDCAST_JSON["hourly"]
    analysis = h["cloud_cover"]
    for n in (1, 5, 7):
        fc = h[f"cloud_cover_previous_day{n}"]
        differing = sum(1 for a, f in zip(analysis, fc)
                        if a is not None and f is not None and a != f)
        assert differing > 0.2 * len(analysis), (
            f"previous_day{n} is suspiciously close to the analysis "
            "— the hindcast would be scoring against its own answer sheet")


def test_score_reports_error_growth_and_a_climatology_baseline():
    scenes = H.fetch_scenes(*SITEA, START, END, search_json=SCENES_JSON)
    s = H.score(H.pair(scenes, HINDCAST_JSON))
    assert set(s) == set(H.LEADS)
    for lead, d in s.items():
        assert d["n"] == len(scenes)
        assert d["mae"] >= 0 and d["rmse"] >= d["mae"]
        assert 0.0 <= d["decision_accuracy"] <= 1.0
        assert 0.0 <= d["brier"] <= 1.0
        assert 0.0 <= d["observed_clear_rate"] <= 1.0
    # forecasts must not get *better* with lead time over a 2-month sample
    assert s[7]["mae"] > s[1]["mae"], "error should grow with lead time"


def test_score_brier_skill_is_relative_to_climatology():
    """A forecast equal to the observed clear-rate every time has zero skill."""
    scenes = H.fetch_scenes(*SITEA, START, END, search_json=SCENES_JSON)
    pairings = H.pair(scenes, HINDCAST_JSON)
    s = H.score(pairings)
    for d in s.values():
        assert d["brier_skill_score"] == pytest.approx(
            1.0 - d["brier"] / d["brier_climatology"], abs=1e-3)


def test_run_is_fully_offline_and_summarises():
    """`run` must make zero network calls when both payloads are injected."""
    def boom(*a, **k):
        raise AssertionError("run() attempted network I/O")

    r = H.run(*SITEA, START, END, http_get=boom, http_post=boom,
              search_json=SCENES_JSON, hindcast_json=HINDCAST_JSON)
    assert r["scenes"] == 20
    assert r["pairings"] == 20 * H.MAX_LEAD_DAYS
    assert sum(r["platforms"].values()) == 20
    assert set(r["by_lead"]) == set(H.LEADS)


def test_run_with_no_scenes_is_graceful():
    r = H.run(*SITEA, START, END, search_json={"features": []},
              hindcast_json=HINDCAST_JSON)
    assert r["scenes"] == 0 and r["pairings"] == 0 and r["by_lead"] == {}
