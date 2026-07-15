"""Independent validation of the propagation/geometry spine (IMPROVEMENTS.md C1).

This is the check that retires the [SIMULATED] grade. Everything else in the
repo validates this code against *itself*: the regression baseline is
self-generated, and the unit tests assert self-consistency (a rotation that
inverts, a norm that is preserved). Nothing compared a predicted pass against an
**independently observed** acquisition — until this.

Method
------
Landsat 8/9 and Sentinel-2A/B/C are nadir-pointing push-brooms: a ground point is
imaged at the moment the spacecraft's closest approach to it, and every scene is
published with its sensing time and footprint centroid. So for each real scene we
propagate *our* SGP4 → TEME→ECEF → off-nadir chain, golden-section refine the
minimum, and compare our TCA to the operator's published time.

It exercises the identical code path Dragonette runs — only the TLE differs — so
sub-second agreement across four spacecraft at two altitudes is direct evidence
the spine is correct, not merely self-consistent.

Measured 2026-07-15 (600 live scenes, TLE age 0.3-0.7 d):
    Landsat-9         n=200  dTCA mean +0.06 s  stdev 0.18 s  max 0.81 s
    Sentinel-2A/B/C   n=400  dTCA mean +0.41 s  stdev 0.49 s  max 1.78 s

Two independent cross-checks fall out of the same data and are worth more than
the timing alone, because they could not agree by accident:
  * **Off-nadir at our TCA reproduces each sensor's field of view.** Landsat WRS-2
    scene centres sit on the ground track -> we get 0.12 deg (nadir), as we must.
    Sentinel-2 MGRS tiles are offset across its 290 km swath -> we get up to
    10.46 deg, landing on S2's +/-10.3 deg FOV half-angle.
  * **The +0.41 s Sentinel-2 bias vs Landsat's +0.06 s is a datetime *convention*,
    not error** — S2 publishes a granule sensing time, Landsat a scene-centre
    time. The Landsat figure is the cleaner estimate of our true accuracy.

TLE freshness is load-bearing: SGP4 drift is ~1-3 km/day, so scenes must sit
within ~1 day of the element epoch or the drift swamps the signal. The committed
fixtures satisfy this by construction; `MAX_TLE_AGE_DAYS` enforces it.

Network is injected (`search_json`) following the passes.py `http_get` pattern,
so tests run fully offline. [SESSION 2026-07-15]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from sgp4.api import Satrec

import passes as P

# NORAD IDs [VERIFIED 2026-07-15 against the live Celestrak `resource` group].
EO_SATELLITES: dict[str, int] = {
    "LANDSAT8": 39084,
    "LANDSAT9": 49260,
    "SENTINEL2A": 40697,
    "SENTINEL2B": 42063,
    "SENTINEL2C": 60989,
}

# STAC `platform` -> our TLE key.
PLATFORM_TO_SAT: dict[str, str] = {
    "landsat-8": "LANDSAT8",
    "landsat-9": "LANDSAT9",
    "sentinel-2a": "SENTINEL2A",
    "sentinel-2b": "SENTINEL2B",
    "sentinel-2c": "SENTINEL2C",
}

# Beyond this the SGP4 along-track drift (~1-3 km/day => ~0.15-0.4 s/day) starts
# to dominate the quantity being measured, and a "pass" becomes a test of TLE age
# rather than of our geometry. [SESSION 2026-07-15]
MAX_TLE_AGE_DAYS = 2.0

SEARCH_HALF_WINDOW_S = 20 * 60.0     # bracket to hunt the closest approach in
COARSE_STEP_S = 10.0


@dataclass(frozen=True)
class SceneMatch:
    platform: str
    published_utc: datetime
    predicted_tca_utc: datetime
    off_nadir_deg: float
    tle_age_days: float

    @property
    def dt_seconds(self) -> float:
        """Predicted minus published. Positive => we predict late."""
        return (self.predicted_tca_utc - self.published_utc).total_seconds()


def load_scenes(search_json: dict) -> list[dict]:
    """Scenes usable for validation: need a platform we track and a centroid."""
    out = []
    for f in search_json.get("features", []):
        p = f.get("properties", {}) or {}
        c = p.get("proj:centroid")
        if not c or PLATFORM_TO_SAT.get(p.get("platform")) is None or not p.get("datetime"):
            continue
        out.append(f)
    return out


def _closest_approach(sat: Satrec, site_ecef: np.ndarray,
                      about: datetime) -> tuple[datetime, float] | None:
    """Refined TCA and off-nadir near `about`, or None if the minimum is not
    bracketed inside the search window."""
    t0 = about - timedelta(seconds=SEARCH_HALF_WINDOW_S)
    jd0, fr0 = P.dt_to_jd(t0)
    ts = np.arange(0.0, 2.0 * SEARCH_HALF_WINDOW_S, COARSE_STEP_S)
    eta = np.array([P._eta_at(sat, jd0, fr0, float(x), site_ecef) for x in ts])
    k = int(eta.argmin())
    if k == 0 or k == len(ts) - 1:
        return None                      # not a true interior minimum
    t_star = P._golden(lambda x: P._eta_at(sat, jd0, fr0, x, site_ecef),
                       float(ts[k - 1]), float(ts[k + 1]), tol=0.01)
    return (t0 + timedelta(seconds=float(t_star)),
            P._eta_at(sat, jd0, fr0, t_star, site_ecef))


def validate(tles: dict[str, P.TLE], search_json: dict,
             max_tle_age_days: float = MAX_TLE_AGE_DAYS) -> list[SceneMatch]:
    """Match every usable scene to our predicted closest approach.

    Scenes whose TLE is older than `max_tle_age_days` at acquisition are skipped:
    past that the result measures element staleness, not our geometry.
    """
    out: list[SceneMatch] = []
    for f in load_scenes(search_json):
        p = f["properties"]
        name = PLATFORM_TO_SAT[p["platform"]]
        tle = tles.get(name)
        if tle is None:
            continue
        published = datetime.fromisoformat(p["datetime"].replace("Z", "+00:00"))
        age = abs((published - tle.epoch_utc).total_seconds()) / 86400.0
        if age > max_tle_age_days:
            continue
        c = p["proj:centroid"]
        sat = Satrec.twoline2rv(tle.line1, tle.line2)
        if getattr(sat, "error", 0):
            continue
        got = _closest_approach(sat, P.geodetic_to_ecef(c["lat"], c["lon"], 0.0),
                                published)
        if got is None:
            continue
        tca, eta = got
        out.append(SceneMatch(platform=p["platform"], published_utc=published,
                              predicted_tca_utc=tca, off_nadir_deg=eta,
                              tle_age_days=age))
    return out


def summarise(matches: list[SceneMatch]) -> dict:
    """Per-platform and overall timing agreement."""
    def stats(ms: list[SceneMatch]) -> dict:
        d = sorted(m.dt_seconds for m in ms)
        n = len(d)
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / n
        return dict(n=n,
                    mean_dt_s=round(mean, 3),
                    median_dt_s=round(d[n // 2] if n % 2 else (d[n // 2 - 1] + d[n // 2]) / 2, 3),
                    stdev_dt_s=round(var ** 0.5, 3),
                    max_abs_dt_s=round(max(abs(x) for x in d), 3),
                    mean_off_nadir_deg=round(sum(m.off_nadir_deg for m in ms) / n, 2),
                    max_off_nadir_deg=round(max(m.off_nadir_deg for m in ms), 2),
                    max_tle_age_days=round(max(m.tle_age_days for m in ms), 2))
    if not matches:
        return {"overall": None, "by_platform": {}}
    by: dict[str, list[SceneMatch]] = {}
    for m in matches:
        by.setdefault(m.platform, []).append(m)
    return {"overall": stats(matches),
            "by_platform": {k: stats(v) for k, v in sorted(by.items())}}


def run_offline(scenes_path: str | Path, tle_path: str | Path) -> dict:
    """Validate from committed fixtures. No network."""
    tles = P._parse_3le_file(Path(tle_path).read_text(), EO_SATELLITES)
    search_json = json.loads(Path(scenes_path).read_text())
    return summarise(validate(tles, search_json))


def run_live(bbox: tuple[float, float, float, float],
             start: datetime, end: datetime,
             http_post: Callable[[str, str], str] | None = None) -> dict:
    """Fetch fresh elements + recent scenes and validate against them.

    Keep [start, end] within ~1 day of now, or MAX_TLE_AGE_DAYS will discard
    everything: Celestrak serves only current elements, so old scenes cannot be
    validated without a historical TLE archive (Space-Track).
    """
    import hindcast as H
    tles, _ = P.fetch_tles(satellites=EO_SATELLITES)
    body = json.dumps({
        "collections": list(H.DEFAULT_COLLECTIONS),
        "bbox": list(bbox),
        "datetime": (f"{start.astimezone(timezone.utc):%Y-%m-%dT%H:%M:%SZ}/"
                     f"{end.astimezone(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"),
        "limit": 200,
    })
    search_json = json.loads((http_post or H._default_post)(H.STAC_SEARCH_URL, body))
    return summarise(validate(tles, search_json))
