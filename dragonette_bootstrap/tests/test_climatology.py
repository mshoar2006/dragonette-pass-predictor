"""Tests for the Tier-3 clear-sky climatology (src/climatology.py).

Tier 3 is what a pass beyond the forecast horizon gets. It shipped as an empty
placeholder, so every such pass returned n/a; it is now estimated from ~1300-1800
real Landsat/Sentinel-2 acquisitions per site. Fully offline.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import climatology as C  # noqa: E402
import hindcast as H  # noqa: E402
import passes as P  # noqa: E402

FIX = ROOT / "fixtures"
SHIPPED = json.loads((ROOT / "src" / "sites_climatology.json").read_text())
MONTHS = set(C.MONTHS)


def _scene(month, cloud, day=1):
    return H.Scene(platform="sentinel-2b", collection="sentinel-2-l2a",
                   acquired_utc=datetime(2024, month, day, 0, 4, tzinfo=timezone.utc),
                   observed_cloud_pct=cloud)


def test_monthly_clear_rate_is_the_fraction_below_threshold():
    scenes = ([_scene(7, 5.0)] * 30 + [_scene(7, 90.0)] * 10)      # 30/40 clear
    rates, counts = C.monthly_clear_rate(scenes, threshold=30.0)
    assert rates["Jul"] == 75.0
    assert counts["Jul"] == 40


def test_rate_moves_with_the_threshold():
    """The rate is meaningless without the threshold — which is why the shipped
    file records it and the test below pins it against the code."""
    scenes = [_scene(7, c) for c in (5.0, 25.0, 45.0, 95.0)] * 10
    lo, _ = C.monthly_clear_rate(scenes, threshold=10.0)
    hi, _ = C.monthly_clear_rate(scenes, threshold=50.0)
    assert lo["Jul"] == 25.0 and hi["Jul"] == 75.0


def test_thin_months_are_omitted_not_published_thin():
    """A month with a handful of scenes must fall through to n/a rather than ship
    a rate built from noise."""
    scenes = [_scene(3, 5.0)] * (C.MIN_SCENES_PER_MONTH - 1) + [_scene(4, 5.0)] * 40
    rates, counts = C.monthly_clear_rate(scenes, threshold=30.0)
    assert "Mar" not in rates and counts["Mar"] == C.MIN_SCENES_PER_MONTH - 1
    assert rates["Apr"] == 100.0


def test_build_is_offline_when_archives_are_supplied():
    def boom(*a, **k):
        raise AssertionError("build() attempted network I/O")
    blob = C.build({"SITE": (-27.0, 151.0)}, threshold=30.0, http_post=boom,
                   archives={"SITE": [_scene(7, 5.0)] * 40})
    assert blob["SITE"]["Jul"] == 100.0
    assert blob["_threshold_pct"] == 30.0
    assert blob["_sample_counts"]["SITE"]["Jul"] == 40


# ------------------------------------------------ the shipped file must stay usable
def test_shipped_climatology_keys_match_real_kmz_polygon_names():
    """THE failure mode this file has: `attach_cloud` looks the AOI up by
    `pred.aoi.name`, so a key that does not exactly match a KMZ polygon name is
    dead weight — the lookup silently returns n/a and nobody notices. The shipped
    placeholder had exactly this problem: it was missing SITEB_2.5ha.
    """
    real = set()
    for f in ("SiteA.kmz", "SiteB.kmz",
              "SiteC.kmz"):
        real |= set(P.list_polygons((FIX / f).read_bytes()))
    keys = {k for k in SHIPPED if not k.startswith("_")}
    assert keys == real, (
        f"climatology keys must match KMZ polygon names exactly.\n"
        f"  missing (would return n/a): {real - keys}\n"
        f"  stale (never looked up):    {keys - real}")


def test_shipped_climatology_threshold_matches_the_code():
    """A clear-sky RATE is defined against a threshold. If CLOUD_OK_THRESHOLD is
    ever tuned (it is an unconfirmed guess — 'tune w/ the mission contact'), this
    file becomes silently wrong. Fail loudly instead: regenerate with
    `python -m climatology`.
    """
    assert SHIPPED["_threshold_pct"] == P.CLOUD_OK_THRESHOLD, (
        f"climatology was built at {SHIPPED['_threshold_pct']}% but "
        f"CLOUD_OK_THRESHOLD is now {P.CLOUD_OK_THRESHOLD}% — regenerate it")


def test_shipped_rates_are_plausible_and_well_formed():
    for k, v in SHIPPED.items():
        if k.startswith("_"):
            continue
        assert v, f"{k} has no months — Tier 3 would be n/a for it"
        assert set(v) <= MONTHS, f"{k} has a bad month key: {set(v) - MONTHS}"
        for m, pct in v.items():
            assert 0.0 <= pct <= 100.0, f"{k}/{m} = {pct} is not a percentage"


def test_shipped_rates_carry_a_real_seasonal_signal():
    """Sanity that this is data and not a constant: at Site A (subtropical QLD)
    the winter dry season must be materially clearer than the summer wet season.
    A flat table would pass every other test here.
    """
    t = SHIPPED["SITEA_100sqkm"]
    assert t["Jul"] > t["Jan"] + 15.0, f"expected a wet/dry contrast, got {t}"


def test_siteC_is_cloudier_than_siteA():
    """Cross-site sanity: coastal South Australia vs subtropical Queensland.
    Independently corroborated by the hindcast harness (mean observed cloud 52%
    at Site C vs 37% at Site A)."""
    mil = SHIPPED["Site C trial site"]
    tos = SHIPPED["SITEA_100sqkm"]
    common = set(mil) & set(tos)
    assert sum(mil[m] for m in common) / len(common) < \
        sum(tos[m] for m in common) / len(common)


def test_tier3_pass_now_gets_a_climatology_instead_of_na():
    """End-to-end: the whole point. Before this, every pass beyond the forecast
    horizon returned clim_clear_pct=None."""
    tles = P._parse_3le_file((FIX / "demo_tles_synthetic.txt").read_text(), P.SATELLITES)
    ref = datetime(2026, 7, 14, tzinfo=timezone.utc)
    pred = P.predict((FIX / "SiteA.kmz").read_bytes(), days=25.0, start_utc=ref,
                     terrain_alt_m=400.0, polygon_name="SITEA_100sqkm", tles=tles,
                     min_sun_elev_deg=-90, marginal_sun_elev_deg=-90,
                     max_off_nadir_deg=60, marginal_off_nadir_deg=60)
    clim = P.load_climatology(ROOT / "src" / "sites_climatology.json")
    P.attach_cloud(pred, climatology=clim, now=ref,
                   http_get=lambda u: (_ for _ in ()).throw(RuntimeError("offline")))
    t3 = [p for p in pred.passes + pred.marginal + pred.nonoperational
          if p.cloud and p.cloud.tier == 3]
    assert t3, "a 25-day window should contain Tier-3 passes"
    assert all(p.cloud.clim_clear_pct is not None for p in t3)
    assert t3[0].cloud.clim_clear_pct == SHIPPED["SITEA_100sqkm"]["Jul"]
