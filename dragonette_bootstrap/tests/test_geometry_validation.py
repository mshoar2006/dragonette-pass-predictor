"""The test that retires simulation: our geometry vs independently observed reality.

Every other test in this repo checks this code against itself. These check it
against Landsat-9 and Sentinel-2B acquisitions that actually happened, using the
operators' own published sensing times and solar geometry.

Fully offline — real captured STAC + Celestrak payloads live in fixtures/.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import geometry_validation as G  # noqa: E402
import passes as P  # noqa: E402

FIX = ROOT / "fixtures"
SCENES = FIX / "stac_scenes_geometry_validation_20260715.json"
TLES = FIX / "tles_landsat_sentinel2_20260715.txt"
SEARCH_JSON = json.loads(SCENES.read_text())


@pytest.fixture(scope="module")
def matches():
    tles = P._parse_3le_file(TLES.read_text(), G.EO_SATELLITES)
    return G.validate(tles, SEARCH_JSON)


def test_fixture_covers_both_constellations(matches):
    plats = {m.platform for m in matches}
    assert any(p.startswith("landsat") for p in plats), plats
    assert any(p.startswith("sentinel-2") for p in plats), plats
    assert len(matches) >= 40, f"expected a meaningful sample, got {len(matches)}"


def test_tles_are_fresh_enough_for_the_measurement_to_mean_anything(matches):
    """SGP4 drifts ~1-3 km/day (~0.15-0.4 s/day). If the elements are stale the
    test silently becomes a measurement of TLE age rather than of our geometry."""
    assert max(m.tle_age_days for m in matches) <= G.MAX_TLE_AGE_DAYS


def test_predicted_tca_matches_published_acquisition_times(matches):
    """THE headline claim. Our SGP4 -> TEME->ECEF -> off-nadir -> golden-section
    chain reproduces when Landsat/Sentinel-2 actually imaged a point, to well
    under a second. Same code path Dragonette runs; only the TLE differs.

    Bounds are ~3x the measured spread (Landsat mean +0.06 s, S2 mean +0.41 s over
    600 live scenes on 2026-07-15), so this fails on a real regression rather than
    on noise.
    """
    s = G.summarise(matches)["overall"]
    assert s["max_abs_dt_s"] < 5.0, f"a pass is mistimed vs reality: {s}"
    assert abs(s["mean_dt_s"]) < 2.0, f"systematic timing bias vs reality: {s}"
    assert s["stdev_dt_s"] < 2.0, f"timing scatter vs reality: {s}"


def test_landsat_agrees_to_sub_second(matches):
    """Landsat is the cleaner probe: WRS-2 scene centres lie on the ground track,
    so there is no cross-track or tile-grid ambiguity, and its published time is a
    scene-centre time. This is our best estimate of true accuracy."""
    ls = [m for m in matches if m.platform.startswith("landsat")]
    assert ls, "fixture should contain Landsat scenes"
    s = G.summarise(ls)["overall"]
    assert abs(s["mean_dt_s"]) < 1.0, f"Landsat timing bias: {s}"
    assert s["max_abs_dt_s"] < 3.0, f"Landsat worst case: {s}"


def test_off_nadir_at_tca_reproduces_each_sensor_field_of_view(matches):
    """An independent shape check the timing cannot fake.

    Landsat scene centres sit on the ground track => off-nadir must be ~0.
    Sentinel-2 MGRS tiles are offset across a 290 km swath => off-nadir spreads
    out to S2's +/-10.3 deg FOV half-angle, and must not exceed it materially.
    """
    ls = [m for m in matches if m.platform.startswith("landsat")]
    s2 = [m for m in matches if m.platform.startswith("sentinel-2")]
    assert max(m.off_nadir_deg for m in ls) < 2.0, \
        "Landsat scene centres should be at nadir"
    assert max(m.off_nadir_deg for m in s2) < 12.0, \
        "Sentinel-2 tiles should fall within its ~10.3 deg FOV half-angle"
    assert max(m.off_nadir_deg for m in s2) > 2.0, \
        "S2 tiles are offset across the swath; all-nadir would mean we lost the geometry"


def test_solar_model_matches_operator_published_sun_geometry():
    """Independent validation of the 2026-07-15 solar rewrite against ESA/USGS
    metadata — not against our own reimplementation.

    Measured over 200 live scenes: elevation mean +0.025 deg, azimuth mean
    +0.027 deg. The pre-fix model scored **azimuth mean +113 deg** here, so this
    test is what would have caught it. Residual spread ~0.1-0.3 deg is the
    tile-mean nature of the published value (the sun varies ~0.5 deg across a
    110 km S2 tile), not our error.
    """
    de, da = [], []
    for f in SEARCH_JSON["features"]:
        p = f["properties"]
        c, se, sa = p.get("proj:centroid"), p.get("view:sun_elevation"), p.get("view:sun_azimuth")
        if not c or se is None or sa is None:
            continue
        t = datetime.fromisoformat(p["datetime"].replace("Z", "+00:00"))
        el, az, _ = P.sun_position_deg(t, c["lat"], c["lon"])
        de.append(el - se)
        da.append((az - sa + 180.0) % 360.0 - 180.0)
    assert len(de) >= 20, f"need a meaningful sample of published sun geometry, got {len(de)}"
    assert abs(sum(de) / len(de)) < 0.3, f"sun elevation bias vs operator metadata: {sum(de)/len(de)}"
    assert max(abs(x) for x in de) < 1.5, "sun elevation outlier vs operator metadata"
    assert abs(sum(da) / len(da)) < 1.0, (
        f"sun azimuth bias vs operator metadata: {sum(da)/len(da)} "
        "(the pre-2026-07-15 model scored ~+113 deg here)")
    assert max(abs(x) for x in da) < 3.0, "sun azimuth outlier vs operator metadata"


def test_stale_tles_are_excluded_rather_than_silently_measured(matches):
    """Guards the guard: with a 0-day tolerance nothing should qualify, proving
    the age filter is real and not vacuous."""
    tles = P._parse_3le_file(TLES.read_text(), G.EO_SATELLITES)
    assert G.validate(tles, SEARCH_JSON, max_tle_age_days=0.0) == []
    assert matches, "but the committed fixture must qualify at the real tolerance"


def test_summarise_is_empty_safe():
    assert G.summarise([]) == {"overall": None, "by_platform": {}}


def test_run_offline_needs_no_network():
    s = G.run_offline(SCENES, TLES)
    assert s["overall"]["n"] >= 40
    assert set(s["by_platform"]) == {"landsat-9", "sentinel-2b"}
