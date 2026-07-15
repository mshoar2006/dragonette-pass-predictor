#!/usr/bin/env python3
"""CLI for the Dragonette pass predictor.

Examples:
  python cli.py SiteA.kmz --days 14 --alt 400 --tz Australia/Brisbane -o siteA.xlsx
  python cli.py site.kmz --tle-file saved_tles.txt          # offline / reproducible
  python cli.py A.kmz B.kmz --all-polygons -o campaign.xlsx # multi-AOI campaign (R8)
"""
import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import passes as P


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict Wyvern Dragonette passes over a KMZ AOI")
    ap.add_argument("kmz", nargs="+",
                    help="One or more KMZ/KML files (multiple => campaign workbook, R8)")
    ap.add_argument("--days", type=float, default=14.0, help="Window length (default 14)")
    ap.add_argument("--start", default=None,
                    help="Window start UTC, ISO format (default: now)")
    ap.add_argument("--alt", type=float, default=0.0, help="Terrain height, metres (default 0)")
    ap.add_argument("--tz", default="Australia/Brisbane",
                    help="IANA timezone for the local-time column")
    ap.add_argument("--max-off-nadir", type=float, default=None,
                    help="Override the sensor's access envelope (default: per --sensor)")
    ap.add_argument("--min-sun", type=float, default=None,
                    help="Override the sensor's sun floor (default: per --sensor)")
    ap.add_argument("--sensor", default="dragonette",
                    help="Sensor profile: dragonette (default, taskable), "
                         "landsat (8/9, fixed nadir), sentinel2 (A/B/C, fixed nadir). "
                         "Landsat/Sentinel-2 are NOT taskable — these are predicted "
                         "acquisitions on their own cycle, not opportunities you can book.")
    ap.add_argument("--polygon", default=None, help="Polygon name filter inside the KMZ")
    ap.add_argument("--all-polygons", action="store_true",
                    help="Predict every polygon in the KMZ into one workbook")
    ap.add_argument("--include-nonoperational", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Include non-operational satellites (DRAG05), shown "
                         "separately and never counted (default: on). "
                         "Use --no-include-nonoperational to drop entirely.")
    ap.add_argument("--tle-file", default=None, help="Use a saved TLE file instead of Celestrak")
    ap.add_argument("--cloud", action="store_true",
                    help="Attach Open-Meteo cloud cover (3-tier; needs network)")
    ap.add_argument("--cloud-threshold", type=float, default=P.CLOUD_OK_THRESHOLD,
                    help="Total-cloud %% counted as clear for Tier-2 P(clear) (default 30)")
    ap.add_argument("--nadir-ellipsoid", action="store_true",
                    help="Measure off-nadir from the WGS84 ellipsoid normal instead of "
                         "geocentric (~0.2 deg difference; off the validated baseline)")
    ap.add_argument("-o", "--out", default=None, help="Output .xlsx path")
    a = ap.parse_args()

    # Validate everything cheap before doing minutes of propagation, and fail
    # with a message rather than a traceback. [SESSION 2026-07-15]
    try:
        profile = P.get_profile(a.sensor)        # raises ValueError on an unknown key
        start = P.parse_start_utc(a.start)          # converts an offset, never relabels it
        P.validate_window(a.days)
        ZoneInfo(a.tz)                              # else we'd only find out after predicting
        if not (0.0 <= a.cloud_threshold <= 100.0):
            raise ValueError(f"--cloud-threshold must be 0-100; got {a.cloud_threshold:g}")
        for path in a.kmz:
            if not Path(path).is_file():
                raise ValueError(f"no such file: {path}")
    except ZoneInfoNotFoundError:
        print(f"error: unknown --tz {a.tz!r} (expected an IANA name, "
              "e.g. Australia/Brisbane)", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def _predict(kmz_bytes, polygon):
        return P.predict(kmz_bytes, days=a.days, start_utc=start, profile=profile,
                         terrain_alt_m=a.alt, max_off_nadir_deg=a.max_off_nadir,
                         min_sun_elev_deg=a.min_sun, polygon_name=polygon,
                         offline_tle_file=a.tle_file,
                         include_nonoperational=a.include_nonoperational,
                         nadir_ellipsoid=a.nadir_ellipsoid)

    # R8: one prediction per (KMZ, polygon); R7 safety applies to each KMZ.
    # Both branches need the same R7 handling — --all-polygons used to escape it
    # and traceback instead. [SESSION 2026-07-15]
    preds = []
    for path in a.kmz:
        kmz_bytes = Path(path).read_bytes()
        try:
            if a.all_polygons:
                names = P.list_polygons(kmz_bytes)
                if not names:
                    print(f"{path}: no polygon found in KMZ/KML", file=sys.stderr)
                    return 2
                preds += [_predict(kmz_bytes, n) for n in names]
            else:
                preds.append(_predict(kmz_bytes, a.polygon))
        except P.AmbiguousPolygonError as exc:
            print(f"{path} contains multiple polygons — choose one with --polygon "
                  "(substring OK) or pass --all-polygons:", file=sys.stderr)
            for n in exc.names:
                print(f"    {n}", file=sys.stderr)
            return 2
        except ValueError as exc:        # bad polygon name / bad KMZ
            print(f"{path}: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:      # Celestrak unreachable and no cache
            print(f"error: {exc}", file=sys.stderr)
            return 3

    if a.cloud:
        clim = P.load_climatology(Path(__file__).with_name("sites_climatology.json"))
        for pred in preds:
            P.attach_cloud(pred, threshold=a.cloud_threshold, climatology=clim)

    default_stem = "campaign" if len(a.kmz) > 1 else Path(a.kmz[0]).stem
    out = Path(a.out or (default_stem + "_passes.xlsx"))
    try:
        out.write_bytes(P.write_xlsx_multi(preds, tz_name=a.tz))
    except OSError as exc:
        print(f"error: cannot write {out}: {exc}", file=sys.stderr)
        return 2

    for pred in preds:
        print(f"AOI: {pred.aoi.name}  centroid {pred.aoi.centroid_lat:.5f}, "
              f"{pred.aoi.centroid_lon:.5f}  (terrain {pred.aoi.terrain_alt_m:.0f} m)")
        print(f"Window: {pred.start_utc:%Y-%m-%d %H:%M}Z + {a.days:g} d")
        print(f"Standard passes: {len(pred.passes)}   Marginal: {len(pred.marginal)}")
        for p in pred.passes:
            print(f"  {p.satellite}  {p.tca_utc:%Y-%m-%dT%H:%M:%SZ}  "
                  f"off-nadir {p.off_nadir_deg:+6.1f}°  sun {p.sun_elev_deg:5.1f}°  "
                  f"slant {p.slant_range_km:.0f} km")
        if pred.nonoperational:
            print(f"Non-operational (not taskable, not counted): "
                  f"{len(pred.nonoperational)} passes")
            for p in pred.nonoperational:
                print(f"  [{p.satellite} NON-OP]  {p.tca_utc:%Y-%m-%dT%H:%M:%SZ}  "
                      f"off-nadir {p.off_nadir_deg:+6.1f}°  sun {p.sun_elev_deg:5.1f}°")
    warned: set[str] = set()
    for pred in preds:
        for w in pred.warnings:
            if w not in warned:
                warned.add(w); print(f"  ! {w}", file=sys.stderr)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
