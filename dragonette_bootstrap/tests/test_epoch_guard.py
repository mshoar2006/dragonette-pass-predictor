"""P1: predict() must reject a window that has drifted far from every loaded
TLE's epoch, and TLE-age timing uncertainty must be symmetric around the epoch.

Two related bugs, both in the A4 timing-uncertainty path:

1. `predict()` had no guard against a window meaning something other than
   "now" -- a year-typo in --start (2024 instead of 2026) run against fresh
   Celestrak elements produced results with nothing louder than a generic
   "oldest TLE is old" warning, easy to miss (the 2024-window incident).
2. The per-pass timing sigma used `max(0.0, age_d)`, so every pre-epoch pass
   reported a flat 0.13 s regardless of how far before the epoch it actually
   was -- SGP4 drift accumulates propagating backward just as it does
   forward.

Fully offline — real captured Celestrak payload in fixtures/.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import passes as P  # noqa: E402

FIX = ROOT / "fixtures"
REAL_TLES = FIX / "tles_real_20260714.txt"
SITEA_KMZ = (FIX / "SiteA.kmz").read_bytes()
POLYGON = "SITEA_100sqkm"


def _tles():
    return P._parse_3le_file(REAL_TLES.read_text(), P.SATELLITES)


# ------------------------------------------------------------- the window guard
def test_year_typo_window_raises_naming_the_satellites():
    """The exact incident: a window two years off from the loaded TLEs'
    epoch. Every satellite in the catalogue is that far off, so the error
    must name them."""
    bad_start = datetime(2024, 7, 14, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="year/date typo") as exc:
        P.predict(SITEA_KMZ, days=3.0, start_utc=bad_start, terrain_alt_m=400.0,
                 polygon_name=POLYGON, tles=_tles())
    msg = str(exc.value)
    assert any(name in msg for name in P.SATELLITES), msg


def test_a_normal_window_at_the_epoch_is_unaffected():
    pred = P.predict(SITEA_KMZ, days=3.0,
                     start_utc=datetime(2026, 7, 14, tzinfo=timezone.utc),
                     terrain_alt_m=400.0, polygon_name=POLYGON, tles=_tles())
    assert pred.passes or pred.marginal or pred.nonoperational  # still works


def test_validate_epoch_window_direct():
    """Unit-level: the guard itself, +/-2 years from a real TLE epoch."""
    tles = _tles()
    epoch = next(iter(tles.values())).epoch_utc
    far_start = epoch - timedelta(days=730)
    with pytest.raises(ValueError, match="year/date typo"):
        P.validate_epoch_window(tles, far_start, far_start + timedelta(days=3))
    # a window that legitimately spans the epoch must pass silently
    P.validate_epoch_window(tles, epoch - timedelta(days=1), epoch + timedelta(days=13))


def test_guard_allows_a_gap_up_to_the_limit_but_not_beyond():
    tles = _tles()
    epoch = next(iter(tles.values())).epoch_utc
    just_inside = epoch + timedelta(days=P.MAX_WINDOW_EPOCH_GAP_DAYS - 1)
    P.validate_epoch_window(tles, just_inside, just_inside + timedelta(days=1))
    just_outside = epoch + timedelta(days=P.MAX_WINDOW_EPOCH_GAP_DAYS + 1)
    with pytest.raises(ValueError):
        P.validate_epoch_window(tles, just_outside, just_outside + timedelta(days=1))


# ------------------------------------------------------- the honest TCA sigma
def test_timing_sigma_is_symmetric_around_the_epoch():
    epoch = datetime(2026, 7, 15, tzinfo=timezone.utc)
    before = epoch - timedelta(days=2)
    after = epoch + timedelta(days=2)
    assert P.timing_sigma_s(before, epoch) == P.timing_sigma_s(after, epoch)
    assert P.timing_sigma_s(before, epoch) == pytest.approx((1.0 + 2.0 * 2.0) / 7.5, abs=0.005)


def test_timing_sigma_no_longer_flattens_pre_epoch_passes():
    """The bug: max(0.0, age_d) clamped every pre-epoch pass to (1.0)/7.5 =
    0.13, regardless of how far before the epoch it actually was."""
    epoch = datetime(2026, 7, 15, tzinfo=timezone.utc)
    flat_bug_value = round(1.0 / 7.5, 2)
    two_days_before = epoch - timedelta(days=2)
    five_days_before = epoch - timedelta(days=5)
    assert P.timing_sigma_s(two_days_before, epoch) != flat_bug_value
    assert P.timing_sigma_s(five_days_before, epoch) > P.timing_sigma_s(two_days_before, epoch)


def test_timing_sigma_at_epoch_is_the_floor():
    epoch = datetime(2026, 7, 15, tzinfo=timezone.utc)
    assert P.timing_sigma_s(epoch, epoch) == pytest.approx(1.0 / 7.5, abs=0.005)
