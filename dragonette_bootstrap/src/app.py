"""FastAPI wrapper for the Dragonette pass predictor.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs   (interactive — upload a KMZ from the browser)

Endpoints are plain `def` (not async): the pipeline is CPU-bound NumPy work,
so FastAPI runs it in its threadpool and the event loop stays free.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

import passes as P

_CLIMATOLOGY = P.load_climatology(Path(__file__).with_name("sites_climatology.json"))

# Bounds the per-request fan-out: each AOI is a full propagation, and with
# cloud=true each also costs Open-Meteo calls.
MAX_UPLOADS = 25


def _maybe_cloud(preds: list[P.Prediction], cloud: bool, threshold: float) -> None:
    if cloud:
        for pred in preds:
            P.attach_cloud(pred, threshold=threshold, climatology=_CLIMATOLOGY)

app = FastAPI(
    title="Dragonette Pass Predictor",
    description="Wyvern Dragonette (DRAG01–05) imaging opportunities over a KMZ AOI. "
                "Geometric access only — tasking availability is separate; cloud is "
                "advisory. JSON contract schema_version 2.1.",
    version="2.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@app.get("/tle-status")
def tle_status() -> dict:
    """Current TLE epochs and their age — check freshness before trusting output."""
    try:
        tles, warnings = P.fetch_tles()
    except Exception as exc:
        raise HTTPException(503, f"TLE fetch failed: {exc}") from exc
    now = datetime.now(timezone.utc)
    return {
        "warnings": warnings,
        "satellites": {
            name: {
                "norad": t.catnr,
                "tle_epoch_utc": t.epoch_utc.isoformat(timespec="seconds"),
                "age_days": round((now - t.epoch_utc).total_seconds() / 86400.0, 2),
                "fetched": t.fetched_utc,
            } for name, t in tles.items()
        },
    }


def _run(kmz: "UploadFile | list[UploadFile]", days: float, alt: float, tz: str,
         max_off_nadir: float, min_sun: float, polygon: str | None,
         all_polygons: bool = False,
         include_nonoperational: bool = True,
         nadir_ellipsoid: bool = False,
         start: str | None = None,
         sensor: str | None = None) -> list[P.Prediction]:
    combined = (sensor or "").strip().lower() in (P.COMBINED_KEY, "combined")
    try:
        if not combined:
            profile = P.get_profile(sensor)   # validates the key
        P.validate_window(days)
        ZoneInfo(tz)                     # fail fast, not after minutes of propagation
        start_utc = P.parse_start_utc(start)
    except ZoneInfoNotFoundError:
        raise HTTPException(422, f"Unknown timezone {tz!r} (expected an IANA name)")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    uploads = kmz if isinstance(kmz, list) else [kmz]     # R8: one or many AOIs

    def one(data: bytes, name: str | None) -> P.Prediction:
        if combined:
            # Predict every constellation for this AOI and merge. Each push-broom
            # uses its OWN native envelope (max_off_nadir/min_sun = None): a fixed
            # nadir sensor cannot roll to the Dragonette 20 deg tasking envelope,
            # so applying it would invent out-of-FOV "opportunities".
            parts = [P.predict(data, days=days, start_utc=start_utc, terrain_alt_m=alt,
                               profile=prof, polygon_name=name,
                               include_nonoperational=include_nonoperational,
                               nadir_ellipsoid=nadir_ellipsoid)
                     for prof in (P.DRAGONETTE, P.LANDSAT, P.SENTINEL2)]
            return P.merge_predictions(parts)
        return P.predict(data, days=days, start_utc=start_utc, terrain_alt_m=alt,
                         profile=profile,
                         max_off_nadir_deg=max_off_nadir, min_sun_elev_deg=min_sun,
                         polygon_name=name,
                         include_nonoperational=include_nonoperational,
                         nadir_ellipsoid=nadir_ellipsoid)
    # /predict is unauthenticated and each AOI costs a full propagation, so bound
    # the fan-out. A KMZ with hundreds of polygons plus cloud=true would otherwise
    # be hours of work and hundreds of Open-Meteo calls in one request.
    if len(uploads) > MAX_UPLOADS:
        raise HTTPException(422, f"Too many files: {len(uploads)}; limit is {MAX_UPLOADS}")

    preds: list[P.Prediction] = []
    try:
        for up in uploads:
            # Read bounded: the limit lives in passes.MAX_UPLOAD_BYTES so the CLI
            # and the API cannot disagree about what is acceptable.
            data = up.file.read(P.MAX_UPLOAD_BYTES + 1)
            if len(data) > P.MAX_UPLOAD_BYTES:
                raise HTTPException(
                    413, f"{up.filename or 'upload'} exceeds "
                         f"{P.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            if not data:
                raise HTTPException(422, f"Empty upload: {up.filename or '?'}")
            if all_polygons:
                names = P.list_polygons(data)
                if not names:
                    raise HTTPException(422, f"No polygon found in {up.filename or 'KMZ'}")
                preds += [one(data, n) for n in names]
            else:
                preds.append(one(data, polygon))
        return preds
    except P.AmbiguousPolygonError as exc:  # R7: never guess — list the choices
        raise HTTPException(422, {"error": str(exc), "polygons": exc.names}) from exc
    except ValueError as exc:               # bad KMZ / polygon
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:             # TLE acquisition failure
        raise HTTPException(503, str(exc)) from exc


@app.post("/predict")
def predict_xlsx(
    kmz: list[UploadFile] = File(..., description="One or more KMZ/KML AOI files"),
    days: float = Form(14.0),
    alt: float = Form(0.0, description="Terrain height, metres"),
    tz: str = Form("Australia/Brisbane", description="IANA tz for local-time column"),
    max_off_nadir: float = Form(20.0),
    min_sun: float = Form(20.0),
    polygon: str | None = Form(None, description="Polygon name filter within the KMZ"),
    all_polygons: bool = Form(False, description="Predict every polygon in the KMZ"),
    include_nonoperational: bool = Form(True, description="Include DRAG05 (non-op), shown separately"),
    cloud: bool = Form(False, description="Attach Open-Meteo cloud cover (3-tier)"),
    cloud_threshold: float = Form(P.CLOUD_OK_THRESHOLD),
    start: str | None = Form(None, description="Window start, ISO-8601. "
                             "An offset is converted to UTC; naive is taken as UTC. "
                             "Default: now."),
    sensor: str = Form("dragonette", description="dragonette | landsat | sentinel2. "
                       "Landsat/Sentinel-2 are fixed nadir push-brooms: predicted "
                       "acquisitions on their own cycle, NOT taskable."),
):
    """Spreadsheet download — the sheet to circulate to research project teams."""
    preds = _run(kmz, days, alt, tz, max_off_nadir, min_sun, polygon, all_polygons,
                 include_nonoperational, start=start, sensor=sensor)
    _maybe_cloud(preds, cloud, cloud_threshold)
    try:
        blob = P.write_xlsx_multi(preds, tz_name=tz)
    except Exception as exc:
        raise HTTPException(422, f"Bad timezone or report error: {exc}") from exc
    stem = ("campaign" if len(kmz) > 1
            else (kmz[0].filename or "aoi").rsplit(".", 1)[0])
    tag = (sensor or "dragonette").strip().lower()
    tag = "allsensors" if tag in (P.COMBINED_KEY, "combined") else tag
    fname = f"{stem}_{tag}_passes_{datetime.now(timezone.utc):%Y%m%d}.xlsx"
    return StreamingResponse(
        io.BytesIO(blob),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.post("/predict/json")
def predict_json(
    kmz: list[UploadFile] = File(...),
    days: float = Form(14.0),
    alt: float = Form(0.0),
    tz: str = Form("Australia/Brisbane"),
    max_off_nadir: float = Form(20.0),
    min_sun: float = Form(20.0),
    polygon: str | None = Form(None),
    all_polygons: bool = Form(False),
    include_nonoperational: bool = Form(True),
    cloud: bool = Form(False),
    cloud_threshold: float = Form(P.CLOUD_OK_THRESHOLD),
    start: str | None = Form(None, description="Window start, ISO-8601. "
                             "An offset is converted to UTC; naive is taken as UTC. "
                             "Default: now."),
    sensor: str = Form("dragonette", description="dragonette | landsat | sentinel2. "
                       "Landsat/Sentinel-2 are fixed nadir push-brooms: predicted "
                       "acquisitions on their own cycle, NOT taskable."),
) -> dict:
    """Same prediction as /predict, machine-readable (DT contract, R9)."""
    preds = _run(kmz, days, alt, tz, max_off_nadir, min_sun, polygon, all_polygons,
                 include_nonoperational, start=start, sensor=sensor)
    _maybe_cloud(preds, cloud, cloud_threshold)
    return P.prediction_json(preds)


@app.post("/timeline.png")
def timeline_png(
    kmz: list[UploadFile] = File(..., description="One or more KMZ/KML AOI files"),
    days: float = Form(14.0),
    alt: float = Form(0.0),
    tz: str = Form("Australia/Brisbane"),
    max_off_nadir: float = Form(20.0),
    min_sun: float = Form(20.0),
    polygon: str | None = Form(None),
    all_polygons: bool = Form(False),
    include_nonoperational: bool = Form(True),
    cloud: bool = Form(False),
    cloud_threshold: float = Form(P.CLOUD_OK_THRESHOLD),
    start: str | None = Form(None, description="Window start, ISO-8601. "
                             "An offset is converted to UTC; naive is taken as UTC. "
                             "Default: now."),
    sensor: str = Form("dragonette", description="dragonette | landsat | sentinel2. "
                       "Landsat/Sentinel-2 are fixed nadir push-brooms: predicted "
                       "acquisitions on their own cycle, NOT taskable."),
):
    """Gantt-style timeline PNG (R4) — same params as /predict; drop into reports."""
    preds = _run(kmz, days, alt, tz, max_off_nadir, min_sun, polygon, all_polygons,
                 include_nonoperational, start=start, sensor=sensor)
    _maybe_cloud(preds, cloud, cloud_threshold)
    try:
        png = P.render_timeline_png(preds, tz_name=tz)
    except Exception as exc:
        raise HTTPException(500, f"Timeline render failed: {exc}") from exc
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


_INDEX_HTML = (Path(__file__).with_name("index.html")).read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Single-page front-end (upload, predict, timeline, tables, download)."""
    return _INDEX_HTML
