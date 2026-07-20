# Dragonette Pass Predictor



Predicts satellite imaging opportunities over an area of interest (AOI) supplied
as KMZ/KML, and produces a spreadsheet for research teams plus versioned JSON for
digital-twin integration. Built around **Wyvern Dragonette** (DRAG01–05), with
**Landsat 8/9** and **Sentinel-2 A/B/C** as first-class sensors — individually or
all at once.

> **Geometric access only.** Wyvern tasking/scheduling is a separate constraint,
> and cloud cover is advisory (it never filters passes). Re-run on fresh TLEs
> before committing a tasking order — SGP4 timing error grows with TLE age.

## What it does

- **Multi-sensor.** Dragonette (agile/taskable), Landsat 8/9 and Sentinel-2 A/B/C
  (fixed nadir push-brooms), or a combined **All sensors** view on one timeline.
- **Per-pass geometry.** Signed off-nadir angle, sun elevation/azimuth, effective
  GSD, slant range, sun-glint, ascending/descending node, and a swath footprint
  with % AOI coverage.
- **Cloud (optional).** Live Open-Meteo forecast — Tier 1 deterministic, Tier 2 a
  51-member ECMWF ensemble → P(clear), Tier 3 real climatology.
- **Outputs.** An `.xlsx` workbook (passes, marginal, non-operational, method,
  timeline chart) and a versioned JSON contract. A browser SPA with an interactive
  map and timeline, plus a self-contained standalone HTML build (no server).
- **Safety by design.** Multi-polygon KMZs never silently guess; non-operational
  satellites are predicted but shown separately, never as taskable; every output
  carries its TLE epochs and staleness warnings.

## Validation

The propagation/geometry spine is validated against **reality**, not just tested:

- **600 live acquisitions** (Landsat-9 / Sentinel-2): predicted closest-approach
  time vs the operators' published sensing time — Landsat-9 mean **+0.06 s**.
- **Footprint outline** measured against real published Landsat scene geometry
  (~189 km vs the modelled ~188 km); Sentinel-2 swath validated via FOV.
- **Solar model** vs published scene metadata: within **~0.03°**.
- **Orbits/identity** cross-checked against the Celestrak SATCAT.
- **Sign convention** pinned column-for-column against Wyvern's own reference sheet.

Dragonette itself has no public scene archive, so its footprint is validated by
inference — identical code path, different TLE — plus an independently validated
orbit.

## Quick start

The project lives in [`dragonette_bootstrap/`](dragonette_bootstrap/):

    cd dragonette_bootstrap
    pip install -r requirements.txt
    python -m pytest tests/ -q                      # offline suite (needs local test fixtures)

    # CLI — point it at your own KMZ/KML area of interest
    python src/cli.py path/to/your-aoi.kmz \
      --alt 400 --tz Australia/Brisbane --sensor dragonette -o out.xlsx

    # Web app
    cd src && uvicorn app:app --port 8000            # open http://127.0.0.1:8000

## Layout

| Path | Contents |
|------|----------|
| `dragonette_bootstrap/src/` | core physics (`passes.py`), CLI, FastAPI app, SPA |
| `dragonette_bootstrap/tests/` | offline test suite |
| `dragonette_bootstrap/README.md` | full CLI/API/options reference |
