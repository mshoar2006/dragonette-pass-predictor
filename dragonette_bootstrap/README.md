# Dragonette Pass Predictor

Predicts Wyvern Dragonette (DRAG01–05) hyperspectral imaging opportunities
over AOI polygons supplied as KMZ, and produces (a) a spreadsheet for research
project teams and (b) versioned JSON for digital-twin integration.

**Geometric access only.** Wyvern tasking/scheduling is a separate constraint,
and cloud cover is advisory (never filters passes). Always re-run on fresh TLEs
before committing a tasking order — SGP4 timing error grows with TLE age.

## Install

    pip install -r requirements.txt
    python -m pytest tests/ -q            # offline suite (needs local test fixtures)

## CLI

    python src/cli.py <kmz...> [options] -o out.xlsx

    python src/cli.py path/to/your-aoi.kmz --polygon <POLYGON_NAME> \
      --alt 400 --tz Australia/Brisbane -o out.xlsx

| Flag | Meaning |
|------|---------|
| `kmz...` | one or more KMZ/KML files (2+ → combined **campaign** workbook, R8) |
| `--days N` | window length, days (default 14) |
| `--start ISO` | window start UTC (default: now) |
| `--alt M` | terrain height, metres |
| `--tz IANA` | timezone for the local-time column (e.g. `Australia/Brisbane`) |
| `--max-off-nadir` / `--min-sun` | access envelope (default 20° / 20°) |
| `--polygon NAME` | pick one polygon by substring (see multi-polygon safety) |
| `--all-polygons` | predict **every** polygon in each KMZ into one workbook |
| `--include-nonoperational` / `--no-include-nonoperational` | show DRAG05 (default on, separated) or drop it |
| `--cloud` | attach Open-Meteo cloud cover (3-tier; needs network) |
| `--cloud-threshold %` | total-cloud % counted as "clear" for Tier-2 P(clear) (default 30) |
| `--nadir-ellipsoid` | measure off-nadir from the WGS84 ellipsoid normal (≈0.2° from the validated geocentric baseline; opt-in) |
| `--tle-file PATH` | use a saved TLE/3LE file instead of Celestrak (offline/reproducible) |
| `-o PATH` | output `.xlsx` |

**Multi-polygon safety (R7).** Many KMZs carry several polygons
(`SiteA.kmz` = `AOI 1`, `AOI 2`, `SITEA_100sqkm`). If a KMZ has >1 polygon
and you give neither `--polygon` nor `--all-polygons`, the tool **lists the
names and exits non-zero** rather than silently guessing.

**DRAG05 (R5).** NORAD 66694 is not yet operational. It is predicted but shown
on a separate, badged **Non-operational** sheet, excluded from headline counts,
and styled distinctly on the timeline. Commissioning is a one-line flip:
`OPERATIONAL["DRAG05"] = True` in `src/passes.py`.

## API (FastAPI)

    cd src && uvicorn app:app --host 0.0.0.0 --port 8000
    # browser form at /   ·   interactive docs at /docs

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | liveness |
| `GET /tle-status` | current TLE epochs and age — check freshness |
| `POST /predict` | `.xlsx` download (Passes / Marginal / Non-operational / Method / Timeline) |
| `POST /predict/json` | machine-readable prediction (DT contract, R9) |
| `POST /timeline.png` | Gantt-style timeline PNG for reports (R4) |

Form fields (multipart) for the three `POST` endpoints: `kmz` (one or more
files), `days`, `alt`, `tz`, `max_off_nadir`, `min_sun`, `polygon`,
`all_polygons`, `include_nonoperational`, `cloud`, `cloud_threshold`.
An ambiguous KMZ returns **422** with `{"error": ..., "polygons": [...]}`.

    # examples
    curl -F kmz=@your-aoi.kmz -F polygon=<POLYGON_NAME> \
         -F alt=400 http://localhost:8000/predict -o out.xlsx
    curl -F kmz=@a.kmz -F kmz=@b.kmz -F all_polygons=true \
         http://localhost:8000/predict/json          # multi-AOI campaign

## JSON contract (schema_version 2.0)

Single AOI → flat top level; multiple AOIs → nested under `aois`. Always carries
`schema_version`. Consumers should branch on presence of `aois`.

    {
      "schema_version": "2.0",
      "aoi": {"name","centroid_lat","centroid_lon","terrain_alt_m"},
      "window_utc": ["<start ISO>", "<end ISO>"],
      "passes":         [ <pass>, ... ],   // operational, standard filter
      "marginal":       [ <pass>, ... ],   // operational, stretch band
      "nonoperational": [ <pass>, ... ],   // DRAG05 etc. — never counted
      "warnings": ["..."],
      "params": { ... echo of run parameters ... }
    }
    // multiple AOIs:  {"schema_version":"2.0","aois":[ {<the above minus schema_version>}, ... ]}

    <pass> = {
      "satellite": "DRAG01",
      "tca_utc": "2026-07-14T23:41:07",
      "off_nadir_deg": -12.3,          // signed, Wyvern convention
      "sun_elev_deg": 27.9,
      "max_off_nadir_aoi_deg": 13.1,   // worst vertex across the AOI
      "slant_range_km": 812,
      "tle_epoch_utc": "2026-07-13T05:12:44",
      "category": "standard",          // "standard" | "marginal"
      "operational": true,
      "cloud": {                        // present only when --cloud/cloud=true
        "tier": 1,                      // 1 forecast | 2 outlook | 3 climatology | 0 n/a
        "label": "forecast",
        "total_pct": 42, "low_pct": 10, "mid_pct": 20, "high_pct": 30,
        "p_clear": null, "threshold_pct": 30, "spread_pct": null,
        "clim_clear_pct": null, "likely_cloudy": false
      }
    }

Contract stability is pinned by tests in `tests/test_v2.py` (R9): adding a key
is a minor change; removing/renaming one is breaking.

## Timeline (R4)

Gantt-style chart, one row per satellite, bars at pass times (widened to a
legible ±3 h), dates on the x-axis. Standard (blue), marginal (amber) and
non-operational (grey, hatched) are visually distinct. Rendered with matplotlib,
embedded as a **Timeline** sheet in every workbook and served at
`POST /timeline.png`. Multiple AOIs share one combined campaign timeline.

## Cloud cover (R6, three tiers)

Skill decays with lead time, so the tier is keyed off lead time from window start:

| Lead | Tier | Source | Shown |
|------|------|--------|-------|
| 0–5 d | 1 | Open-Meteo forecast | deterministic total + low/mid/high %, label "forecast" |
| 5–15 d | 2 | Open-Meteo ensemble | P(cloud < threshold) + member spread, "outlook (probabilistic)" |
| >15 d | 3 | climatology | per-site clear-sky base rate (`src/sites_climatology.json`) |

One forecast call + one ensemble call **per AOI** (batched). Degrades gracefully:
on any failure the cloud columns read `n/a`, a warning is added, and the
prediction never blocks. Attribution "Weather data by Open-Meteo.com" (CC BY 4.0)
is written to the Method sheet. Tier-3 climatology is real observed data,
checked against the live source, until the VG26003 clr% values are supplied.

## Live validation (blocking release)

Offline tests use synthetic TLEs and simulated Open-Meteo samples. Before
any output is shared beyond the dev machine, run the one-time live check on a
networked machine and record the results:

    python fetch_real_data.py        # real Celestrak TLEs + real Open-Meteo, compared to reference

A restricted/CI sandbox cannot reach Celestrak/Open-Meteo (network allowlist), so
this is a developer-machine step. Until it passes, all outputs are simulated-grade,
not yet confirmed against live data.

## Contents

- `src/passes.py` — framework-free core (physics, cloud, timeline, xlsx, JSON)
- `src/cli.py`, `src/app.py` — thin CLI + FastAPI wrappers
- `src/sites_climatology.json` — Tier-3 base rates (placeholder)
- `tests/` — offline suite (`test_all.py` v1 baseline + `test_v2.py` R4–R9). Runs
  against local test fixtures (AOI KMZs, TLEs, captured API payloads) that are kept
  out of this repo; supply your own to run the fixture-dependent tests.
- `fetch_real_data.py` — one-command live-data + validation script
