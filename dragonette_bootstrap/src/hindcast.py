"""Cloud-forecast hindcast harness (IMPROVEMENTS.md C1 — validation loop).

Answers one question with measured numbers rather than assertion: **how well does
the Open-Meteo cloud forecast this tool relies on actually do over our AOIs?**

Method
------
Landsat 8/9 and Sentinel-2A/B/C fly sun-synchronous mid-morning orbits very close
to Dragonette's (all overfly our AU sites within ~45 min of each other in local
solar time), and every one of their scenes is published with an observed cloud
percentage. That makes them a free, dense, retrospective proxy for "what was the
sky actually doing when a Dragonette-like sensor flew over?" — thousands of
labelled samples, versus the handful Wyvern's own archive can supply.

For each real acquisition we ask Open-Meteo what it *would have forecast* for that
timestamp at lead times of 1-7 days, and compare against the scene's own observed
cloud. That yields error-vs-lead-time curves, and the decision-level question the
tool actually poses: at the operational threshold, would we have called it right?

The load-bearing subtlety
--------------------------
The historical-forecast API's plain `cloud_cover` is NOT a forecast at lead time.
Measured over 10 days at Site A it is **240/240 hours identical to the ERA5
reanalysis** — it is the analysis, i.e. the answer sheet. Scoring against it would
report near-perfect skill and mean nothing. A true hindcast requires the
`cloud_cover_previous_dayN` fields, which carry the forecast issued N days before
the valid time and diverge from the analysis exactly as forecasts should. N is
capped at 7 (day 8+ returns an empty series), so Tier 1 (0-5 d) is fully
measurable, Tier 2 (5-15 d) only over days 5-7, and days 7-15 not at all by this
route.

Provenance per DEVELOPMENT.md: this endpoint behaviour was exercised against the
live APIs. Both are keyless. Network access is injected
(`http_get`/`http_post`) following the `passes.py` `http_get` pattern, so the
tests run fully offline.

Caveat: `eo:cloud_cover` is a **whole-scene** statistic — a
Sentinel-2 tile is ~110x110 km and a Landsat scene ~185x180 km, against AOIs of
2.5 ha-100 km2. It is therefore a noisy proxy for cloud over the AOI itself, and
biases toward the regional mean. Per-AOI truth needs the scene's own cloud mask
(S2 SCL / Landsat QA_PIXEL) clipped to the polygon; that is a rasterio/COG job and
is deliberately out of scope here. Treat these numbers as regional skill, not
per-paddock skill.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

# Free, keyless, no auth. Checked that /collections lists
# sentinel-2-l2a and landsat-c2-l2.
STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
HINDCAST_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Collections carrying an observed per-scene cloud percentage (`eo:cloud_cover`).
DEFAULT_COLLECTIONS = ("sentinel-2-l2a", "landsat-c2-l2")

# Open-Meteo serves forecasts issued at most 7 days before the valid time.
# Checked that previous_day7 returns data; previous_day8/10 do not.
MAX_LEAD_DAYS = 7
LEADS = tuple(range(1, MAX_LEAD_DAYS + 1))

UA = {"User-Agent": "dragonette-predictor/hindcast (research)"}

# A scene is matched to the forecast hour nearest its acquisition. Both series are
# hourly, so a genuine match is always within 30 min; anything beyond this means
# the series does not cover the scene and must not be snapped. Mirrors
# passes._CLOUD_MAX_SNAP_H — same silent-carry-over failure mode.
MAX_SNAP_H = 1.5


@dataclass(frozen=True)
class Scene:
    """One real acquisition with its observed cloud percentage."""
    platform: str
    collection: str
    acquired_utc: datetime
    observed_cloud_pct: float


@dataclass(frozen=True)
class Pairing:
    """A scene joined to what was forecast for it `lead_days` ahead."""
    scene: Scene
    lead_days: int
    forecast_cloud_pct: float

    @property
    def error(self) -> float:
        """Forecast minus observed; positive = forecast too cloudy."""
        return self.forecast_cloud_pct - self.scene.observed_cloud_pct


def _default_post(url: str, body: str) -> str:
    import requests
    r = requests.post(url, headers={**UA, "Content-Type": "application/json"},
                      data=body, timeout=60)
    r.raise_for_status()
    return r.text


def _default_get(url: str) -> str:
    import requests
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.text


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fetch_scenes(lat: float, lon: float, start: datetime, end: datetime,
                 collections: tuple[str, ...] = DEFAULT_COLLECTIONS,
                 limit: int = 100,
                 http_post: Callable[[str, str], str] | None = None,
                 search_json: dict | None = None) -> list[Scene]:
    """Real acquisitions intersecting (lat, lon) in [start, end].

    Supply `search_json` to run offline. Scenes without an `eo:cloud_cover` are
    dropped: a scene with no observed cloud cannot be scored against.
    """
    if search_json is None:
        body = json.dumps({
            "collections": list(collections),
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": (f"{start.astimezone(timezone.utc):%Y-%m-%dT%H:%M:%SZ}/"
                         f"{end.astimezone(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"),
            "limit": limit,
        })
        search_json = json.loads((http_post or _default_post)(STAC_SEARCH_URL, body))

    out: list[Scene] = []
    for f in search_json.get("features", []):
        p = f.get("properties", {}) or {}
        cloud = p.get("eo:cloud_cover")
        when = p.get("datetime")
        if cloud is None or not when:
            continue
        out.append(Scene(platform=str(p.get("platform", "unknown")),
                         collection=str(f.get("collection", "unknown")),
                         acquired_utc=_parse_iso(when),
                         observed_cloud_pct=float(cloud)))
    out.sort(key=lambda s: s.acquired_utc)
    return out


def hindcast_url(lat: float, lon: float, start: datetime, end: datetime,
                 leads: tuple[int, ...] = LEADS) -> str:
    fields = ["cloud_cover"] + [f"cloud_cover_previous_day{n}" for n in leads]
    return (f"{HINDCAST_BASE}?latitude={lat}&longitude={lon}"
            f"&start_date={start.astimezone(timezone.utc):%Y-%m-%d}"
            f"&end_date={end.astimezone(timezone.utc):%Y-%m-%d}"
            f"&hourly={','.join(fields)}&timezone=UTC")


def fetch_hindcast(lat: float, lon: float, start: datetime, end: datetime,
                   leads: tuple[int, ...] = LEADS,
                   http_get: Callable[[str], str] | None = None,
                   hindcast_json: dict | None = None) -> dict:
    """Archived forecasts for [start, end] at each lead. One call for all leads."""
    if any(n < 1 or n > MAX_LEAD_DAYS for n in leads):
        raise ValueError(f"leads must be within 1..{MAX_LEAD_DAYS}; got {leads}")
    if hindcast_json is None:
        url = hindcast_url(lat, lon, start, end, leads)
        hindcast_json = json.loads((http_get or _default_get)(url))
    return hindcast_json


def _series(hindcast_json: dict, key: str) -> tuple[list[datetime], list]:
    h = hindcast_json.get("hourly") or {}
    times = [_parse_iso(t) for t in h.get("time", [])]
    return times, h.get(key) or []


def pair(scenes: list[Scene], hindcast_json: dict,
         leads: tuple[int, ...] = LEADS,
         max_snap_h: float = MAX_SNAP_H) -> list[Pairing]:
    """Join each scene to the forecast issued `lead` days before it.

    A scene outside the returned series, or whose lead value is null, is skipped
    rather than snapped to the nearest available hour — the same failure this
    project already hit once in `attach_cloud`.
    """
    out: list[Pairing] = []
    for n in leads:
        times, vals = _series(hindcast_json, f"cloud_cover_previous_day{n}")
        if not times or not vals:
            continue
        for s in scenes:
            i = min(range(len(times)),
                    key=lambda i: abs((times[i] - s.acquired_utc).total_seconds()))
            if abs((times[i] - s.acquired_utc).total_seconds()) > max_snap_h * 3600.0:
                continue
            if i >= len(vals) or vals[i] is None:
                continue
            out.append(Pairing(scene=s, lead_days=n,
                               forecast_cloud_pct=float(vals[i])))
    return out


def score(pairings: list[Pairing], threshold: float = 30.0) -> dict:
    """Skill by lead time.

    `threshold` is the operational clear/cloudy cut (passes.CLOUD_OK_THRESHOLD).
    Reports both the continuous error (bias/MAE/RMSE of cloud %) and the
    decision the tool actually makes: would we have called clear/cloudy right?

    `brier` scores the binary "clear" event, converting the deterministic forecast
    to a probability with the same logistic ramp `attach_cloud` uses for ensemble
    membership, so the number reflects this tool's own decision rule rather than
    an abstract one. Baseline for comparison is `brier_climatology`, i.e. always
    predicting the observed clear-rate — a model with no skill scores the same.
    """
    by_lead: dict[int, list[Pairing]] = {}
    for p in pairings:
        by_lead.setdefault(p.lead_days, []).append(p)

    out: dict[int, dict] = {}
    for lead in sorted(by_lead):
        ps = by_lead[lead]
        n = len(ps)
        errs = [p.error for p in ps]
        obs_clear = [1.0 if p.scene.observed_cloud_pct < threshold else 0.0 for p in ps]
        base_rate = sum(obs_clear) / n
        # p(clear) from the deterministic %, logistic ramp width 8 (passes.py:w)
        p_clear = [1.0 / (1.0 + math.exp((p.forecast_cloud_pct - threshold) / 8.0))
                   for p in ps]
        brier = sum((f - o) ** 2 for f, o in zip(p_clear, obs_clear)) / n
        brier_clim = sum((base_rate - o) ** 2 for o in obs_clear) / n
        hits = sum(1 for f, o in zip(p_clear, obs_clear) if (f >= 0.5) == (o == 1.0))
        out[lead] = dict(
            n=n,
            bias=round(sum(errs) / n, 2),
            mae=round(sum(abs(e) for e in errs) / n, 2),
            rmse=round(math.sqrt(sum(e * e for e in errs) / n), 2),
            observed_clear_rate=round(base_rate, 3),
            decision_accuracy=round(hits / n, 3),
            brier=round(brier, 4),
            brier_climatology=round(brier_clim, 4),
            # >0 means we beat "always predict the base rate"; <=0 means no skill.
            brier_skill_score=round(1.0 - brier / brier_clim, 3) if brier_clim else None,
        )
    return out


def run(lat: float, lon: float, start: datetime, end: datetime,
        threshold: float = 30.0, leads: tuple[int, ...] = LEADS,
        http_get: Callable[[str], str] | None = None,
        http_post: Callable[[str, str], str] | None = None,
        search_json: dict | None = None,
        hindcast_json: dict | None = None) -> dict:
    """Full harness: scenes -> archived forecasts -> paired -> scored.

    Two network calls per AOI (one STAC search, one Open-Meteo request covering
    every lead). Supply search_json/hindcast_json to run fully offline.
    """
    scenes = fetch_scenes(lat, lon, start, end, http_post=http_post,
                          search_json=search_json)
    if not scenes:
        return dict(scenes=0, pairings=0, by_lead={}, platforms={})
    hc = fetch_hindcast(lat, lon, start - timedelta(days=1), end + timedelta(days=1),
                        leads=leads, http_get=http_get, hindcast_json=hindcast_json)
    pairings = pair(scenes, hc, leads=leads)
    platforms: dict[str, int] = {}
    for s in scenes:
        platforms[s.platform] = platforms.get(s.platform, 0) + 1
    return dict(scenes=len(scenes), pairings=len(pairings),
                platforms=dict(sorted(platforms.items())),
                observed_cloud_mean=round(
                    sum(s.observed_cloud_pct for s in scenes) / len(scenes), 2),
                by_lead=score(pairings, threshold))
