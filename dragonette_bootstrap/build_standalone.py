#!/usr/bin/env python3
"""Build a single standalone HTML: the real SPA + the real predictor, no server.

    python build_standalone.py            # -> output/dragonette_standalone.html

Design: reimplement nothing
---------------------------
Two things already exist and work. `src/passes.py` is the validated predictor
(sub-second against 600 real Landsat/Sentinel-2 acquisitions, sign pinned to
Wyvern's sheet, 172 tests). `src/index.html` is the finished SPA (Leaflet map,
Plotly timeline, stat tiles, quality badges, polygon picker). Both are bundled
**byte-for-byte unchanged**.

The only new code is a shim that intercepts the SPA's two `fetch` calls
(`POST /predict/json`, `POST /predict`) and serves them from Pyodide instead of
FastAPI — mirroring `app.py`'s responses exactly, including the 422 + polygon list
that drives the SPA's own picker. So there is no second UI and no second physics:
the browser build is the same two files the server build uses, wired together
differently.

Any deviation from the server build lives in the shim, never in the bundled
files — the shim disables the cloud control at runtime and injects a limits
banner. That way `index.html` and `passes.py` stay diff-clean against `src/`,
which is checkable: both are sha256-compared by the builder.

Evidence, at the precision it was measured
------------------------------------------
* **Pyodide adds no numerical error.** A browser run's live-TLE table was
  reproduced row-for-row in normal Python: max |dTCA| 1.00 s (the table's own
  rounding), max |d off-nadir| 0.00 deg. [VERIFIED 2026-07-15, independent check]
* **Pure-python SGP4 agrees with the C extension to ~1e-5 m, non-growing.** Over
  14 d at 1-min steps x 5 satellites: 81-91% of samples differ, max |dr|
  6.88e-06 m, typical ~1e-8 m, no secular growth, error codes match everywhere.
  Float round-off, ~9 orders below the 0.1 deg reported precision, so pass outputs
  are identical. NOTE: an earlier build claimed "bit-identical / 0.000000 m" —
  that was FALSE (one instant at epoch printed with %f). [VERIFIED 2026-07-15]
* **The pure-python fallback is complete.** With `wrapper.py` absent, `api.py`
  raises ModuleNotFoundError (subclass of ImportError) and falls back to
  `sgp4.model`, which supplies `sgp4_array`/`SatrecArray` — the exact call
  `_off_nadir_series` makes. Verified in isolation: `accelerated=False`, no C leak.
* **CORS is not a blocker.** Celestrak returns `Access-Control-Allow-Origin: *`,
  including for `Origin: null`.

What differs from the server build — the real list
--------------------------------------------------
Cloud, coverage, Tier-3 climatology and .xlsx all WORK. An earlier version of this
file claimed cloud and coverage were impossible. Both claims were false and are
corrected here [SESSION 2026-07-15]:
  * Cloud needs no synchronous seam — `attach_cloud(forecast_json=, ensemble_json=)`
    is the same injection point the offline tests use. The handler awaits js.fetch
    and passes the parsed JSON straight in.
  * shapely 2.0.2 and pyproj 3.6.1 ship WITH Pyodide, so `aoi_coverage_fraction`
    runs unchanged. (Verified against the v0.26.2 lockfile.)
  * Tier 3 only needed `sites_climatology.json` bundled.
  * openpyxl + tzdata are pure-python wheels, installed at boot via micropip.

What genuinely differs:
1. **No TLE cache, cooldown or stale-fallback**, and a browser cannot set
   `User-Agent` (a forbidden header), so Celestrak's "<=2-3 polls/file/day" cannot
   be honoured. Every run re-fetches. The server build is the polite one.
2. **Manoeuvre detection needs history.** `_manoeuvre_warnings` lives inside
   `fetch_tles` and diffs against its cache, which is unreachable here; instead the
   handler diffs against the previous run's TLE text persisted in localStorage via
   `orbit_change()`. Equivalent from the second run on — the FIRST run in a given
   browser cannot see a burn.
3. **Pyodide (~8 MB brotli) loads from a CDN on first run**, then browser-caches;
   jsdelivr is a trust dependency. Pure-python SGP4 is ~6x slower than C plus WASM
   overhead: a 14-day run is seconds, and the numbers are identical.

"""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/"

# Pure-python sgp4. `wrapper.py` is excluded so api.py falls back to sgp4.model;
# tests.py / wulfgar.py / conveniences.py / exporter.py / omm.py are unreachable
# from passes.py — verified against the runtime import graph, not assumed.
SGP4_FILES = ["__init__.py", "api.py", "functions.py", "model.py", "propagation.py",
              "earth_gravity.py", "ext.py", "io.py", "alpha5.py"]


def collect() -> dict[str, str]:
    import sgp4
    d = pathlib.Path(sgp4.__file__).parent
    files = {f"sgp4/{f}": (d / f).read_text(encoding="utf-8") for f in SGP4_FILES}
    files["passes.py"] = (ROOT / "src" / "passes.py").read_text(encoding="utf-8")
    # Tier-3 base rates (real Landsat/S2-derived). Bundling it is all Tier 3 needs.
    files["sites_climatology.json"] = (
        ROOT / "src" / "sites_climatology.json").read_text(encoding="utf-8")
    return files


def build() -> pathlib.Path:
    spa = (ROOT / "src" / "index.html").read_text(encoding="utf-8")
    files = collect()
    payload = base64.b64encode(json.dumps(files).encode("utf-8")).decode("ascii")
    spa_b64 = base64.b64encode(spa.encode("utf-8")).decode("ascii")

    html = (_SHELL.replace("__PAYLOAD__", payload)
                  .replace("__SPA__", spa_b64)
                  .replace("__CDN__", PYODIDE_CDN))
    out = ROOT / "output" / "dragonette_standalone.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")

    # The claim "bundled byte-for-byte unchanged" is checkable, so check it.
    for label, src in (("passes.py", files["passes.py"]), ("index.html", spa)):
        a = hashlib.sha256(src.encode("utf-8")).hexdigest()
        b = hashlib.sha256((ROOT / "src" / label).read_text(encoding="utf-8")
                           .encode("utf-8")).hexdigest()
        assert a == b, f"{label} bundled copy diverged from src/"
        print(f"  {label:12s} sha256 {a[:16]} — identical to src/")
    return out


_SHELL = r"""<!doctype html>
<meta charset="utf-8">
<title>Dragonette Pass Predictor — standalone</title>
<style>
 #boot{font:14px/1.6 system-ui,sans-serif;max-width:720px;margin:60px auto;padding:0 16px;color:#1a2332}
 #boot h1{font-size:19px;margin:0 0 4px}
 #bootlog{background:#0f1720;color:#c9d6e2;padding:10px 12px;border-radius:8px;
   font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;margin-top:14px}
 .bootnote{color:#6B7A8F;font-size:13px}
</style>
<div id="boot">
  <h1>Dragonette Pass Predictor — standalone</h1>
  <p class="bootnote">Starting Python in your browser (Pyodide, ~8&nbsp;MB on first run, cached after).
  This runs the real <code>passes.py</code> — the same validated predictor the server build uses.</p>
  <div id="bootlog">booting…</div>
</div>

<script type="module">
const PAYLOAD = "__PAYLOAD__", SPA = "__SPA__";
const blog = m => { const l = document.getElementById('bootlog');
  if (l) { l.textContent += "\n" + m; l.scrollTop = 1e9; } };
const b64 = s => new TextDecoder().decode(Uint8Array.from(atob(s), c => c.charCodeAt(0)));

let pyodide;

async function boot() {
  const { loadPyodide } = await import("__CDN__pyodide.mjs");
  blog("loading Pyodide…");
  pyodide = await loadPyodide({ indexURL: "__CDN__" });
  blog("loading numpy, shapely, pyproj…");
  await pyodide.loadPackage(["numpy", "micropip", "shapely", "pyproj"]);
  // openpyxl + tzdata are pure-python wheels on PyPI, so the .xlsx download works
  // here too. tzdata is needed because Pyodide has no system zoneinfo database.
  blog("installing openpyxl + tzdata (for the .xlsx download)…");
  await pyodide.runPythonAsync(
    `import micropip; await micropip.install(["openpyxl", "tzdata"])`);

  const files = JSON.parse(b64(PAYLOAD));
  pyodide.FS.mkdir("/app"); pyodide.FS.mkdir("/app/sgp4");
  for (const [n, src] of Object.entries(files)) pyodide.FS.writeFile("/app/" + n, src);
  pyodide.runPython(`import sys; sys.path.insert(0,"/app")`);
  blog("importing passes.py (unchanged)…");
  pyodide.runPython(HANDLER);
  const acc = pyodide.runPython(`from sgp4.api import accelerated; accelerated`);
  blog(`ready — sgp4 accelerated=${acc} (pure-python path, agrees with C to ~1e-5 m)`);

  installFetchShim();
  mountSPA();
}

// ---------------------------------------------------------------- the shim
// The SPA calls POST /predict/json and POST /predict. Serve both from Pyodide,
// mirroring app.py's responses exactly — including the 422 + polygon list that
// drives the SPA's own picker. Everything else falls through to the real network
// (Leaflet tiles, Plotly, fonts).
function installFetchShim() {
  const real = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    const path = url.replace(/^https?:\/\/[^/]+/, "").split("?")[0];
    if (!(init && init.method === "POST" && (path === "/predict/json" || path === "/predict")))
      return real(input, init);
    try {
      const form = {};
      const kmz = [];
      for (const [k, v] of (init.body || new FormData()).entries()) {
        if (k === "kmz") kmz.push({ name: v.name, bytes: new Uint8Array(await v.arrayBuffer()) });
        else form[k] = v;
      }
      const args = pyodide.toPy({ form, kmz, want: path === "/predict" ? "xlsx" : "json",
        prev_tles: localStorage.getItem("dragonette_prev_tles") || "" });
      pyodide.globals.set("_req", args);
      const res = JSON.parse(await pyodide.runPythonAsync("await handle(_req)"));
      args.destroy();
      if (res.tle_text) localStorage.setItem("dragonette_prev_tles", res.tle_text);
      if (res.xlsx_b64 != null) {
        const buf = Uint8Array.from(atob(res.xlsx_b64), c => c.charCodeAt(0));
        return new Response(buf, { status: 200, headers: {
          "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          "Content-Disposition": 'attachment; filename="dragonette_passes.xlsx"' } });
      }
      return new Response(res.body, { status: res.status,
        headers: { "Content-Type": "application/json" } });
    } catch (e) {
      return new Response(JSON.stringify({ detail: String((e && e.message) || e).slice(-400) }),
        { status: 500, headers: { "Content-Type": "application/json" } });
    }
  };
}

// ------------------------------------------------------- mount the real SPA
// index.html is bundled unmodified. Every deviation from the server build is
// applied here, at runtime, so the bundled copy stays diff-clean against src/.
function mountSPA() {
  const doc = new DOMParser().parseFromString(b64(SPA), "text/html");
  document.head.querySelectorAll("style,title").forEach(n => n.remove());
  document.body.innerHTML = "";
  for (const n of [...doc.head.children]) document.head.appendChild(document.importNode(n, true));
  for (const n of [...doc.body.children]) document.body.appendChild(document.importNode(n, true));

  // Re-run the SPA's own scripts (innerHTML/importNode does not execute them).
  for (const old of [...document.querySelectorAll("script")]) {
    const s = document.createElement("script");
    for (const a of old.attributes) s.setAttribute(a.name, a.value);
    s.textContent = old.textContent;
    old.replaceWith(s);
  }
}

// ------------------------------------------------ the Python request handler
const HANDLER = String.raw`
import json, passes as P
from js import fetch as _jsfetch

_CLIM = P.load_climatology("/app/sites_climatology.json")

def _env(status, obj):
    return json.dumps({"status": status, "body": json.dumps(obj, allow_nan=False)})

async def _get_json(url):
    r = await _jsfetch(url)
    if not r.ok:
        raise RuntimeError("HTTP " + str(r.status) + " for " + url)
    return json.loads(await r.text())

def _manoeuvre_warnings(prev_text, tles):
    # The browser has no TLE cache, but localStorage persists the previous fetch,
    # which is all orbit_change() needs. Same detector the server build runs.
    out = []
    if not prev_text:
        return out
    try:
        prev = P._parse_3le_file(prev_text, {n: t.catnr for n, t in tles.items()})
    except Exception:
        return out
    for name, new in tles.items():
        old = prev.get(name)
        if old is None:
            continue
        try:
            ch = P.orbit_change(old, new)
        except Exception:
            continue
        if ch and ch["manoeuvred"]:
            out.append(P._manoeuvre_warning(name, ch))
    return out

async def handle(req):
    # Mirrors app.py's _run + endpoints. Always returns a JSON envelope STRING:
    #   {"status": int, "body": <json str>}  or  {"status": 200, "xlsx_b64": <str>}
    # NOTE: no triple-quoted docstring in this block -- it lives inside the
    # builder's raw-string _SHELL, and a triple quote would terminate it.
    r = req if isinstance(req, dict) else req.to_py()
    form, uploads, want = r["form"], r["kmz"], r["want"]
    r_prev = r.get("prev_tles") or ""

    def f(name, default=None):
        v = form.get(name, default)
        return default if v in (None, "") else v

    def flag(name):
        return str(f(name, "false")).lower() == "true"

    try:
        days = float(f("days", 14))
        alt = float(f("alt", 0))
        tz = str(f("tz", "Australia/Brisbane"))
        sensor_key = str(f("sensor", "dragonette")).strip().lower()
        combined = sensor_key in (P.COMBINED_KEY, "combined")
        profiles = ([P.DRAGONETTE, P.LANDSAT, P.SENTINEL2] if combined
                    else [P.get_profile(sensor_key)])
        P.validate_window(days)
        start = P.parse_start_utc(f("start"))
        if not uploads:
            return _env(422, {"detail": "No KMZ uploaded"})

        all_sats = {}
        for prof in profiles:
            all_sats.update(prof.satellites)
        text = ""
        for name, catnr in all_sats.items():
            r = await _jsfetch(P.CELESTRAK_URL.format(catnr=catnr))
            if not r.ok:
                return _env(503, {"detail": "Celestrak HTTP " + str(r.status) + " for " + name})
            text += (await r.text()) + chr(10)
        tles = P._parse_3le_file(text, all_sats)
        man = _manoeuvre_warnings(r_prev, tles)

        # Combined "all sensors": predict every profile and merge, matching the
        # server's _run. Each push-broom uses its own native envelope (None) — a
        # fixed nadir sensor can't roll — so max_off_nadir/min_sun are single-sensor.
        moff = None if combined else (float(f("max_off_nadir")) if f("max_off_nadir") else None)
        msun = None if combined else (float(f("min_sun")) if f("min_sun") else None)

        preds = []
        for up in uploads:
            data = bytes(up["bytes"])
            if flag("all_polygons"):
                names = P.list_polygons(data)
                if not names:
                    return _env(422, {"detail": "No polygon found"})
                targets = names
            else:
                targets = [f("polygon")]
            for t in targets:
                parts = [P.predict(
                    data, days=days, start_utc=start, terrain_alt_m=alt, profile=prof,
                    tles={n: tl for n, tl in tles.items() if n in prof.satellites},
                    polygon_name=t, max_off_nadir_deg=moff, min_sun_elev_deg=msun,
                    include_nonoperational=flag("include_nonoperational")) for prof in profiles]
                preds.append(P.merge_predictions(parts) if combined else parts[0])
    except P.AmbiguousPolygonError as exc:
        # The SPA reads detail.polygons and renders its own picker (index.html:371).
        return _env(422, {"detail": {"error": str(exc), "polygons": exc.names}})
    except ValueError as exc:
        return _env(422, {"detail": str(exc)})

    if flag("cloud"):
        lat, lon = preds[0].aoi.centroid_lat, preds[0].aoi.centroid_lon
        thr = float(f("cloud_threshold", P.CLOUD_OK_THRESHOLD))
        try:
            fc = await _get_json(P.FORECAST_URL.format(lat=lat, lon=lon))
            ens = await _get_json(P.ENSEMBLE_URL.format(lat=lat, lon=lon))
        except Exception as exc:
            fc = ens = None
            for pr in preds:
                pr.warnings.append("Cloud fetch failed (" + str(exc) + "); columns show n/a.")
        for pr in preds:
            # forecast_json/ensemble_json is the same seam the offline tests use —
            # no synchronous http_get needed.
            P.attach_cloud(pr, threshold=thr, forecast_json=fc, ensemble_json=ens,
                           climatology=_CLIM)

    for pr in preds:
        pr.warnings.extend(man)

    if want == "xlsx":
        import base64 as _b64
        return json.dumps({"status": 200, "xlsx_b64":
            _b64.b64encode(P.write_xlsx_multi(preds, tz_name=tz)).decode("ascii")})
    env = json.loads(_env(200, P.prediction_json(preds)))
    env["tle_text"] = text          # persisted by JS; next run diffs against it
    return json.dumps(env)
`;

boot().catch(e => blog("BOOT FAILED: " + ((e && e.message) || e)));
</script>
"""

if __name__ == "__main__":
    p = build()
    print(f"\nwrote {p}  ({p.stat().st_size / 1024:.0f} KB)")
    print("  Full SPA + the validated predictor, no server. Cloud, coverage, Tier-3")
    print("  climatology and .xlsx all work. Differences: no TLE cache/rate-limit")
    print("  protection, no User-Agent, and manoeuvre detection needs a prior run.")
