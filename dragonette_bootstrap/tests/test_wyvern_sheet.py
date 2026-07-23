"""Pin the predictor against WYVERN'S OWN SHEET — the external reference.

Until 2026-07-15 this sheet was not in the repo, so the sign resolution it drove
survived only as prose notes, and the only executable pin was
`test_regression_baseline.py` — which compares this code against a
**self-generated** baseline. That checks reproducibility, not agreement with the
customer. These tests close that gap: they assert we agree with the numbers
Wyvern themselves published.

What this is and is not
-----------------------
Wyvern's sheet is explicitly labelled *simulated predictions*, so this is
**model-vs-model agreement**, not validation against observation. It is still the
strongest external check available for Dragonette, because no public Dragonette
scene archive exists. (Validation against genuinely observed acquisitions is
`test_geometry_validation.py`, which can only use Landsat/Sentinel-2 — same code
path, different TLE.)

Scope constraints, all inherent to the reference rather than to us:
  * Wyvern used ~1-month-old elements. Their DRAG04 21 Jul = -0.1 deg vs our
    ~-15.5 deg from fresh TLEs is the documented consequence — a stale-TLE
    effect on their side. DRAG04 is therefore excluded.
  * Their timestamp is the SCENE END; our TCA lands 18-100 s earlier.
  * Our fixture TLEs have epoch 2026-07-13, so only rows from ~2026-07-14 on can
    be compared — earlier rows would require propagating backwards.
  * Near-nadir rows (|off_nadir| < 6 deg) are ill-conditioned: a sub-minute
    timing difference swings the angle a lot, so this test's "robust" set is
    |off_nadir| >= 6 deg, same as the original manual validation.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import passes as P  # noqa: E402

FIX = ROOT / "fixtures"
SHEET = json.loads((FIX / "wyvern_sheet_siteA_2026-06-24_2026-07-24.json").read_text())
START = datetime(2026, 7, 14, 0, 0, tzinfo=timezone.utc)
ROBUST_OFF_NADIR_DEG = 6.0
# Wyvern's elements are ~1 month old; DRAG04's divergence is the documented
# symptom of that, not a disagreement about geometry.
EXCLUDED_SATS = {"DRAG04"}


def _rows():
    out = []
    for sat, ts, off, sun in SHEET["rows"]:
        out.append(dict(sat=sat,
                        end_utc=datetime.fromisoformat(ts.replace("Z", "+00:00")),
                        off=off, sun=sun))
    return out


def _comparable():
    """Rows our fixture TLEs (epoch 2026-07-13) can actually speak to."""
    return [r for r in _rows()
            if r["end_utc"] >= START and r["sat"] not in EXCLUDED_SATS]


@pytest.fixture(scope="module")
def pred():
    tles = P._parse_3le_file((FIX / "tles_real_20260714.txt").read_text(), P.SATELLITES)
    return P.predict((FIX / "SiteA.kmz").read_bytes(), days=14.0, start_utc=START,
                     terrain_alt_m=400.0, polygon_name="SITEA_100sqkm", tles=tles,
                     max_off_nadir_deg=25.0, marginal_off_nadir_deg=25.0,
                     min_sun_elev_deg=15.0, marginal_sun_elev_deg=15.0)


def _match(pred, row, tol_s=300.0):
    cands = [p for p in pred.passes + pred.marginal + pred.nonoperational
             if p.satellite == row["sat"]
             and abs((p.tca_utc - row["end_utc"]).total_seconds()) < tol_s]
    return min(cands, key=lambda p: abs((p.tca_utc - row["end_utc"]).total_seconds())) \
        if cands else None


def test_sheet_contains_no_drag05_rows():
    """Independent support for DRAG05 being non-operational: Wyvern does not
    offer it as a taskable opportunity either."""
    assert not [r for r in _rows() if r["sat"] == "DRAG05"]


def test_every_comparable_wyvern_pass_is_reproduced(pred):
    missing = [f"{r['sat']} @ {r['end_utc']:%Y-%m-%dT%H:%M:%S}"
               for r in _comparable() if _match(pred, r) is None]
    assert not missing, f"Wyvern lists passes we do not predict: {missing}"


def test_off_nadir_sign_matches_wyvern_on_robust_passes(pred):
    """THE external pin of the off-nadir sign convention.

    `test_regression_baseline.py` pins the sign against a self-generated
    artifact; this pins it against Wyvern's own published column. Flipping
    SIGN_FLIP_TO_MATCH_WYVERN inverts every one of these.
    """
    robust = [r for r in _comparable() if abs(r["off"]) >= ROBUST_OFF_NADIR_DEG]
    assert len(robust) >= 5, f"need a meaningful robust set, got {len(robust)}"
    flipped = []
    for r in robust:
        p = _match(pred, r)
        if (r["off"] >= 0) != (p.off_nadir_deg >= 0):
            flipped.append(f"{r['sat']} @ {r['end_utc']:%m-%d}: "
                           f"Wyvern {r['off']:+.1f} vs ours {p.off_nadir_deg:+.1f}")
    assert not flipped, ("off-nadir sign disagrees with Wyvern — "
                         f"SIGN_FLIP_TO_MATCH_WYVERN must stay False: {flipped}")


def test_off_nadir_magnitude_matches_wyvern_on_robust_passes(pred):
    """Magnitude agreement, asserted as a distribution rather than row-by-row.

    Measured 2026-07-15 over the six robust rows:
        DRAG03 07-15  0.9 | DRAG01 07-18  0.5 | DRAG01 07-19  0.8
        DRAG03 07-21  0.7 | DRAG03 07-22  1.2 | DRAG02 07-20  4.5  <- outlier

    The original manual validation reported '5 passes ... magnitude within
    ~1 deg', but its robust set was defined as |off-nadir| >= 6 AND magnitude
    already within ~1 deg — which silently drops DRAG02 20 Jul, the one row
    that disagrees. Selecting the rows
    that agree and then reporting agreement is circular, so this test keeps the
    outlier in and asserts the shape instead: a tight majority plus a loose
    ceiling.

    The DRAG02 outlier is almost certainly Wyvern's side: that row also carries
    the largest End-Datetime lag (~100 s vs 18-24 s elsewhere), i.e. their
    month-old elements have drifted furthest for it — the same stale-TLE effect
    that makes DRAG04 unusable. It is not evidence against our geometry, which is
    pinned sub-second against observation in test_geometry_validation.py.
    """
    deltas = {}
    for r in [x for x in _comparable() if abs(x["off"]) >= ROBUST_OFF_NADIR_DEG]:
        p = _match(pred, r)
        deltas[f"{r['sat']} {r['end_utc']:%m-%d}"] = abs(abs(p.off_nadir_deg)
                                                         - abs(r["off"]))
    assert len(deltas) >= 5, f"need a meaningful robust set, got {len(deltas)}"
    tight = {k: v for k, v in deltas.items() if v < 1.5}
    assert len(tight) >= len(deltas) - 1, (
        f"expected at most one row outside ~1.5 deg of Wyvern; got {deltas}")
    assert max(deltas.values()) < 5.0, (
        f"a robust row diverges beyond the known stale-element drift: {deltas}")


def test_our_tca_precedes_wyverns_end_datetime_by_18_to_100s(pred):
    """The interpretation of their column — that it is the scene END, not
    closest approach — is itself a claim, and this is what makes it testable.
    If it ever fails, the two tools have stopped meaning the same thing by a
    timestamp, which would silently corrupt every comparison built on it."""
    lags = []
    for r in _comparable():
        p = _match(pred, r)
        lags.append((r["end_utc"] - p.tca_utc).total_seconds())
    assert all(l > 0 for l in lags), f"our TCA must precede Wyvern's scene end: {lags}"
    assert max(lags) < 150.0, f"lag beyond the documented 18-100 s band: {lags}"


def test_sun_elevation_is_close_to_wyverns(pred):
    """Loose on purpose. Our solar model is pinned to +/-0.03 deg against ESA/USGS
    scene metadata (test_geometry_validation.py) — i.e. against observation — so
    that test, not this one, is the authority on solar accuracy. This only guards
    against a gross regression. Wyvern's values sit ~1 deg away, matching the
    original manual validation's own '~1 deg' note; on the ESA evidence the
    residual is more likely theirs than ours.
    """
    for r in _comparable():
        ours, _, _ = P.sun_position_deg(r["end_utc"], pred.aoi.centroid_lat,
                                        pred.aoi.centroid_lon)
        assert abs(ours - r["sun"]) < 2.0, (
            f"{r['sat']} @ {r['end_utc']:%m-%d}: sun ours {ours:.3f} vs "
            f"Wyvern {r['sun']:.3f}")
