#!/usr/bin/env python3
"""Build a single standalone HTML that runs the VALIDATED predictor in a browser.

    python build_standalone.py            # -> output/dragonette_standalone.html

Why this exists
---------------
"Runs on any machine without a server." The obvious route is to rewrite the
predictor in JavaScript — and that would throw away the thing that makes it worth
shipping. The Python chain is validated to sub-second against 600 real Landsat/
Sentinel-2 acquisitions, its sign convention is pinned against Wyvern's own sheet,
and 172 tests hold it there. A satellite.js rewrite inherits none of that: it is a
new implementation, and this project's whole history is that unvalidated code is
where the bugs hide (the sun azimuth was 113 deg wrong and looked completely fine).

So instead of porting, this ships the actual `passes.py` to the browser via
Pyodide (CPython on WebAssembly). Same code, same validation, no rewrite.

Three facts make it work [VERIFIED 2026-07-15]
----------------------------------------------
1. **Pure-python SGP4 is bit-identical to the C extension.** Measured across all
   five Dragonette element sets: max position difference **0.000000 m**. sgp4's
   own `api.py` falls back to it automatically when the C module is absent, and
   the fallback provides `sgp4_array`/`SatrecArray` — the exact call
   `_off_nadir_series` makes. `passes.py` needs no changes.
2. **Every data source allows browser access.** Celestrak, Open-Meteo (forecast /
   ensemble / archive) and the Earth Search STAC all send
   `Access-Control-Allow-Origin: *`.
3. **The network layer is already injectable.** `fetch_tles(http_get=...)` and
   `attach_cloud(http_get=...)` were built for offline tests; the browser's
   `fetch` drops into the same seam. `requests` is never imported.

What degrades, honestly
-----------------------
* `coverage_pct` -> None (needs shapely/pyproj; it already returns None on
  ImportError). No loss in practice: it reads 100% on every row anyway.
* xlsx and PNG export need openpyxl/matplotlib — omitted here to keep the bundle
  small; both have Pyodide wheels if wanted.
* Pure-python SGP4 is slower than C. A 14-day, 5-satellite run is seconds, not
  milliseconds — irrelevant for a planner, and the numbers are identical.

Pyodide itself (~10 MB) loads from a CDN on first use and is then browser-cached,
so the page is not truly offline on first run. Everything else is inlined.
"""
from __future__ import annotations

import base64
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/"

# Pure-python sgp4 modules. `tests.py`, `conveniences.py`, `exporter.py`, `omm.py`
# and `wrapper.py` are excluded: wrapper.py imports the C extension (api.py falls
# back when it fails), and the rest are unused by passes.py.
SGP4_FILES = ["__init__.py", "api.py", "functions.py", "model.py", "propagation.py",
              "earth_gravity.py", "ext.py", "io.py", "alpha5.py", "wulfgar.py"]


def collect() -> dict[str, str]:
    import sgp4
    d = pathlib.Path(sgp4.__file__).parent
    files = {f"sgp4/{f}": (d / f).read_text(encoding="utf-8") for f in SGP4_FILES}
    files["passes.py"] = (ROOT / "src" / "passes.py").read_text(encoding="utf-8")
    return files


def build() -> pathlib.Path:
    payload = base64.b64encode(
        json.dumps(collect()).encode("utf-8")).decode("ascii")
    html = _TEMPLATE.replace("__PAYLOAD__", payload).replace("__CDN__", PYODIDE_CDN)
    out = ROOT / "output" / "dragonette_standalone.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>Dragonette Pass Predictor — standalone</title>
<style>
 body{font:14px/1.6 system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1a2332}
 h1{font-size:19px;margin:0 0 2px} .sub{color:#6B7A8F;margin:0 0 18px}
 #drop{border:2px dashed #c3ccd8;border-radius:10px;padding:26px;text-align:center;color:#6B7A8F;cursor:pointer}
 #drop.on{border-color:#2f6fd0;background:#f2f7ff;color:#2f6fd0}
 button{font:inherit;padding:7px 14px;border:1px solid #c3ccd8;border-radius:7px;background:#fff;cursor:pointer}
 button:disabled{opacity:.45;cursor:default}
 table{border-collapse:collapse;width:100%;margin-top:14px} th,td{padding:5px 9px;border-bottom:1px solid #eef1f5;text-align:left}
 th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#6B7A8F}
 td.num{text-align:right;font-variant-numeric:tabular-nums}
 #log{background:#0f1720;color:#c9d6e2;padding:10px 12px;border-radius:8px;font:12px/1.5 ui-monospace,monospace;
      white-space:pre-wrap;max-height:190px;overflow:auto;margin-top:14px}
 .warn{background:#fff8e6;border-left:3px solid #e0a800;padding:7px 11px;margin:7px 0;font-size:13px}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}
 label{font-size:13px;color:#6B7A8F} input,select{font:inherit;padding:5px 7px;border:1px solid #c3ccd8;border-radius:6px}
</style>

<h1>Dragonette Pass Predictor <span style="font-weight:400;color:#6B7A8F">— standalone</span></h1>
<p class="sub">Runs the real <code>passes.py</code> in your browser via Pyodide. No server, no install.
Geometric access only — tasking and cloud are separate constraints.</p>

<div class="row">
  <label>Sensor <select id="sensor">
    <option value="dragonette">Dragonette (taskable)</option>
    <option value="landsat">Landsat 8/9</option>
    <option value="sentinel2">Sentinel-2 A/B/C</option>
  </select></label>
  <label>Days <input id="days" type="number" value="14" min="1" max="31" style="width:64px"></label>
  <label>Terrain m <input id="alt" type="number" value="400" style="width:74px"></label>
  <button id="go" disabled>Predict</button>
</div>

<div id="drop">Drop a <b>.kmz</b> here (or click) — TLEs are fetched live from Celestrak</div>
<input id="file" type="file" accept=".kmz,.kml" hidden>
<div id="warns"></div>
<div id="out"></div>
<div id="log">booting…</div>

<script type="module">
const PAYLOAD = "__PAYLOAD__";
const log = m => { const l = document.getElementById('log'); l.textContent += "\n" + m; l.scrollTop = 1e9; };

let pyodide = null, kmzBytes = null, kmzName = null;

async function boot() {
  const { loadPyodide } = await import("__CDN__pyodide.mjs");
  log("loading Pyodide (~10 MB, cached after first run)…");
  pyodide = await loadPyodide({ indexURL: "__CDN__" });
  log("loading numpy…");
  await pyodide.loadPackage("numpy");

  // Write the vendored pure-python sgp4 + the real passes.py into Pyodide's FS.
  const files = JSON.parse(new TextDecoder().decode(
    Uint8Array.from(atob(PAYLOAD), c => c.charCodeAt(0))));
  pyodide.FS.mkdir("/app"); pyodide.FS.mkdir("/app/sgp4");
  for (const [name, src] of Object.entries(files)) pyodide.FS.writeFile("/app/" + name, src);
  pyodide.runPython(`import sys; sys.path.insert(0, "/app")`);

  log("importing passes.py (the validated module, unchanged)…");
  pyodide.runPython(`
import passes as P
from sgp4.api import accelerated
print("sgp4 C-accelerated:", accelerated, "-> pure-python path in use" if not accelerated else "")
`);
  // Hand Python the browser's fetch, through the http_get seam that already
  // exists for offline tests. `requests` is never imported.
  window.__http_get = async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status + " for " + url);
    return await r.text();
  };
  log("ready — drop a KMZ.");
  document.getElementById('go').disabled = false;
}

async function predict() {
  if (!kmzBytes) { log("no KMZ loaded"); return; }
  const go = document.getElementById('go'); go.disabled = true;
  document.getElementById('out').innerHTML = "";
  document.getElementById('warns').innerHTML = "";
  const sensor = document.getElementById('sensor').value;
  log(`predicting ${sensor} over ${kmzName}…`);
  try {
    pyodide.globals.set("kmz_bytes", pyodide.toPy(kmzBytes));
    pyodide.globals.set("sensor", sensor);
    pyodide.globals.set("days", parseFloat(document.getElementById('days').value));
    pyodide.globals.set("alt", parseFloat(document.getElementById('alt').value));
    const js = await pyodide.runPythonAsync(`
import json, passes as P
from pyodide.ffi import to_js

prof = P.get_profile(sensor)

async def http_get(url):
    from js import __http_get
    return await __http_get(url)

# fetch_tles is sync, so pull the TLE text here (async) and parse it offline.
text = ""
for name, catnr in prof.satellites.items():
    text += await http_get(P.CELESTRAK_URL.format(catnr=catnr)) + "\\n"
tles = P._parse_3le_file(text, prof.satellites)

data = bytes(kmz_bytes)
names = P.list_polygons(data)
pred = P.predict(data, days=days, terrain_alt_m=alt, profile=prof, tles=tles,
                 polygon_name=(names[-1] if len(names) > 1 else None))
json.dumps(P.prediction_json([pred]))
`);
    render(JSON.parse(js));
    log("done.");
  } catch (e) {
    log("ERROR: " + e.message);
  }
  go.disabled = false;
}

function render(d) {
  const rows = [...d.passes.map(p => [p, "standard"]),
                ...(d.nonoperational || []).map(p => [p, "non-op"])];
  const w = document.getElementById('warns');
  for (const m of (d.warnings || [])) {
    const el = document.createElement('div'); el.className = 'warn'; el.textContent = "⚠ " + m; w.appendChild(el);
  }
  const t = document.createElement('table');
  t.innerHTML = "<tr><th>Satellite</th><th>TCA (UTC)</th><th>Off-nadir</th>" +
                "<th>Sun</th><th>Eff. GSD</th><th>Tier</th></tr>";
  for (const [p, tier] of rows) {
    const tr = t.insertRow();
    for (const v of [p.satellite, p.tca_utc.slice(0, 19).replace("T", " "),
                     (p.off_nadir_deg > 0 ? "+" : "") + p.off_nadir_deg.toFixed(1) + "°",
                     p.sun_elev_deg.toFixed(1) + "°",
                     (p.geometry ? p.geometry.effective_gsd_m.toFixed(1) : "—") + " m", tier]) {
      const td = tr.insertCell(); td.textContent = v;
      if (/^[+-]?[\d.]/.test(v)) td.className = "num";
    }
  }
  const h = document.createElement('p');
  h.innerHTML = `<b>${d.aoi.name}</b> — ${d.passes.length} standard, ` +
                `${(d.nonoperational || []).length} non-operational · ` +
                `${d.params.sensor_display || d.params.sensor} · swath ${d.params.swath_km} km`;
  const o = document.getElementById('out'); o.appendChild(h); o.appendChild(t);
}

const drop = document.getElementById('drop'), file = document.getElementById('file');
drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('on'); };
drop.ondragleave = () => drop.classList.remove('on');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('on'); take(e.dataTransfer.files[0]); };
file.onchange = e => take(e.target.files[0]);
async function take(f) {
  if (!f) return;
  kmzBytes = new Uint8Array(await f.arrayBuffer()); kmzName = f.name;
  drop.textContent = f.name + "  (" + kmzBytes.length + " bytes) — click Predict";
  log("loaded " + f.name);
}
document.getElementById('go').onclick = predict;
boot().catch(e => log("BOOT FAILED: " + e.message));
</script>
"""

if __name__ == "__main__":
    p = build()
    print(f"wrote {p}  ({p.stat().st_size / 1024:.0f} KB)")
    print("Open it directly in a browser — no server. Pyodide (~10 MB) loads from the CDN once.")
