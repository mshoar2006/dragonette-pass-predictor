"""Tier-3 clear-sky climatology, estimated from real acquisitions (C3).

Tier 3 is the >15 day cloud tier: beyond any forecast horizon, all we can honestly
offer is "historically, how often is this site clear at Dragonette's overpass?".
`sites_climatology.json` shipped as an empty placeholder awaiting a VG26003
analysis from Wyvern, so every pass beyond 15 days returned n/a.

It does not need to wait. Landsat 8/9 and Sentinel-2A/B/C have overflown these
sites mid-morning for a decade and published an observed cloud percentage for
every scene — ~110 observations per site per month over 2016-2025. That is the
"VG26003-style clr%" C3 specifies, and it is free.

Why observed scenes rather than ERA5
-------------------------------------
Open-Meteo's ERA5 archive can also produce this, and is denser (any hour, any
site). But measured against the scene archive at Site A over 2016-2025, restricted
to 9-11 h local solar, ERA5 is **systematically ~6 pp pessimistic** (mean -5.9 pp,
worst month -17.0 pp). The two disagree because they measure different things:
ERA5 reports a ~31 km model column fraction including thin cirrus, while
`eo:cloud_cover` is the sensor's own cloud mask — and it is the sensor's verdict
that decides whether a scene is usable. So the scene archive is primary here;
ERA5 remains a reasonable fallback for a site with no scene history, provided the
bias is remembered.

Threshold dependence — IMPORTANT
--------------------------------
A "clear-sky rate" is only meaningful against a cloud threshold, and this uses
`passes.CLOUD_OK_THRESHOLD` — a sensible default (30%) that is user-adjustable at
request time. **If that threshold changes, this file is wrong and must be
regenerated** — hence `_threshold_pct` is written into the output and asserted by
tests. Note the threshold is a decision line, not a weather statistic: scene data
tells you how cloudy, the threshold decides how cloudy is "too cloudy".

Caveat: `eo:cloud_cover` is a whole-scene statistic (S2 tile ~110x110 km) against
AOIs of 2.5 ha-100 km2, so this is a *regional* base rate, not per-paddock.
Sharpening it needs the scene cloud mask clipped to the polygon (S2 SCL /
Landsat QA_PIXEL).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import hindcast as H

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DEFAULT_START_YEAR = 2016
DEFAULT_END_YEAR = 2025

# Below this many observations a monthly rate is too noisy to publish; emit None
# so `attach_cloud` shows n/a rather than a number built from a handful of scenes.
MIN_SCENES_PER_MONTH = 20


def fetch_archive(lat: float, lon: float,
                  start_year: int = DEFAULT_START_YEAR,
                  end_year: int = DEFAULT_END_YEAR,
                  http_post: Callable[[str, str], str] | None = None
                  ) -> list[H.Scene]:
    """Every catalogued Landsat/Sentinel-2 acquisition over a point, by year.

    Paginated per year because the STAC search caps out (limit=300 returns 502;
    200 is safe) and a decade over one point is ~1000 scenes.
    """
    out: list[H.Scene] = []
    for yr in range(start_year, end_year + 1):
        out += H.fetch_scenes(
            lat, lon,
            datetime(yr, 1, 1, tzinfo=timezone.utc),
            datetime(yr, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            limit=200, http_post=http_post)
    return out


def monthly_clear_rate(scenes: list[H.Scene], threshold: float,
                       min_scenes: int = MIN_SCENES_PER_MONTH
                       ) -> tuple[dict[str, float], dict[str, int]]:
    """(clear-sky % by month, observation count by month).

    A month with fewer than `min_scenes` observations is omitted rather than
    published thin — the caller then falls through to n/a.
    """
    by_month: dict[int, list[float]] = defaultdict(list)
    for s in scenes:
        by_month[s.acquired_utc.month].append(s.observed_cloud_pct)
    rates: dict[str, float] = {}
    counts: dict[str, int] = {}
    for m in range(1, 13):
        vals = by_month.get(m, [])
        counts[MONTHS[m - 1]] = len(vals)
        if len(vals) < min_scenes:
            continue
        rates[MONTHS[m - 1]] = round(
            100.0 * sum(1 for v in vals if v < threshold) / len(vals), 1)
    return rates, counts


def build(sites: dict[str, tuple[float, float]], threshold: float,
          start_year: int = DEFAULT_START_YEAR,
          end_year: int = DEFAULT_END_YEAR,
          http_post: Callable[[str, str], str] | None = None,
          archives: dict[str, list[H.Scene]] | None = None) -> dict:
    """Build the whole sites_climatology.json payload.

    `sites` maps the AOI polygon name **exactly as it appears in the KMZ** (that
    is what `attach_cloud` keys on) to (lat, lon). Supply `archives` to build
    offline from already-fetched scenes.
    """
    blob: dict = {
        "_note": ("Per-site monthly clear-sky rate (%), the Tier-3 cloud "
                  "base rate. Key = AOI polygon name exactly as it appears in the "
                  "KMZ; inner key = 3-letter month. Regenerate with "
                  "`python -m climatology` (needs network). Months with fewer than "
                  f"{MIN_SCENES_PER_MONTH} observations are omitted -> lookup returns n/a."),
        "_provenance": ("Derived from observed Landsat 8/9 + "
                        "Sentinel-2A/B/C acquisitions over each AOI centroid "
                        f"({start_year}-{end_year}), via the Element84 Earth Search STAC "
                        "catalogue; rate = fraction of scenes with eo:cloud_cover below "
                        "the threshold. This is the 'VG26003-style clr%' C3 specifies, "
                        "computed from the public archive rather than supplied. NOT "
                        "Wyvern-confirmed — replace if/when the VG26003 analysis arrives."),
        "_threshold_pct": threshold,
        "_caveats": [
            "THRESHOLD-DEPENDENT: rates are the fraction of scenes below _threshold_pct. "
            "If passes.CLOUD_OK_THRESHOLD changes, this file is stale and must be "
            "regenerated. The threshold is a user-adjustable default (30%), not a weather "
            "statistic — it is the clear/cloudy decision line.",
            "REGIONAL, NOT PER-PADDOCK: eo:cloud_cover is a whole-scene statistic (S2 tile "
            "~110x110 km) against AOIs of 2.5 ha-100 km2. Sharpening needs the scene cloud "
            "mask clipped to the polygon (S2 SCL / Landsat QA_PIXEL).",
            "Sampled at Landsat/Sentinel-2 overpass (~10 h local solar), which matches 14 "
            "of 17 Dragonette passes (9-11 h). Dragonette's ~15 h descending-node passes "
            "are NOT represented and cloud has a diurnal cycle.",
            "Cross-checked against ERA5 over Site A 2016-2025 restricted to 9-11 h local "
            "solar: ERA5 is systematically ~6 pp pessimistic (mean -5.9, worst -17.0), so "
            "the two are not interchangeable.",
        ],
        "_schema": {"<AOI polygon name>": {"Jan": 0.0, "Feb": 0.0, "Jul": 0.0}},
        "_sample_counts": {},
    }
    for name, (lat, lon) in sites.items():
        scenes = (archives or {}).get(name)
        if scenes is None:
            scenes = fetch_archive(lat, lon, start_year, end_year, http_post=http_post)
        rates, counts = monthly_clear_rate(scenes, threshold)
        blob[name] = rates
        blob["_sample_counts"][name] = counts
    return blob


def _sites_from_fixtures(root: Path) -> dict[str, tuple[float, float]]:
    import passes as P
    out: dict[str, tuple[float, float]] = {}
    for f in ("SiteA.kmz", "SiteB.kmz",
              "SiteC.kmz"):
        data = (root / "fixtures" / f).read_bytes()
        for name in P.list_polygons(data):
            aoi = P.parse_kmz(data, 0.0, polygon_name=name)
            out[name] = (aoi.centroid_lat, aoi.centroid_lon)
    return out


if __name__ == "__main__":                       # python -m climatology  (needs network)
    import passes as P
    root = Path(__file__).resolve().parents[1]
    sites = _sites_from_fixtures(root)
    print(f"building Tier-3 climatology for {len(sites)} polygons, "
          f"threshold {P.CLOUD_OK_THRESHOLD:g}% ...")
    blob = build(sites, P.CLOUD_OK_THRESHOLD)
    out = Path(__file__).with_name("sites_climatology.json")
    out.write_text(json.dumps(blob, indent=2))
    for name in sites:
        n = sum(blob["_sample_counts"][name].values())
        print(f"  {name:38s} {len(blob[name]):2d}/12 months from {n:4d} scenes")
    print(f"wrote {out}")
