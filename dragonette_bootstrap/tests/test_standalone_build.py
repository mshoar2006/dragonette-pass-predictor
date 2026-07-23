"""Tests for the standalone browser build (build_standalone.py).

Every test here exists because that check was MISSING and the bug shipped. The
first standalone was demoed live, looked perfect, and got signed off as
verified — a thorough follow-up review found ~20 defects, several of them the
project's signature failure (confidently-wrong output). Three of those failures
were build-time-detectable and would have been caught by the parse checks below:

  * a Python comment containing a backtick silently terminated the JS template
    literal and killed the whole page;
  * a Python docstring's triple quote terminated the builder's own raw string;
  * an escaping slip produced an unterminated string literal in the embedded
    Python, which only surfaced as a browser BOOT FAILED.

These are offline and fast: they check the ARTIFACT, not a browser. Driving the
page is a separate, manual step.
"""
import ast
import base64
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

build_standalone = pytest.importorskip("build_standalone")


@pytest.fixture(scope="module")
def html():
    return build_standalone.build().read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def handler_src(html):
    m = re.search(r"const HANDLER = String\.raw`(.*?)`;", html, re.S)
    assert m, "the embedded Python handler block is missing from the build"
    return m.group(1)


@pytest.fixture(scope="module")
def bundle(html):
    m = re.search(r'const PAYLOAD = "([^"]+)"', html)
    assert m, "payload missing"
    return json.loads(base64.b64decode(m.group(1)).decode("utf-8"))


# ---------------------------------------------------- the three that bit tonight
def test_embedded_python_is_valid_python(handler_src):
    """A bad escape produced `text += (await r.text()) + "` — an unterminated
    string literal that only surfaced in the browser as BOOT FAILED."""
    ast.parse(handler_src)


def test_embedded_python_contains_no_backticks(handler_src):
    """The handler lives inside a JS template literal. One backtick in a Python
    comment silently terminated it and killed the page."""
    assert "`" not in handler_src


def test_builder_shell_contains_no_triple_quotes():
    """_SHELL is a Python raw triple-quoted string; a triple quote inside it (e.g.
    a Python docstring in the handler) terminates the builder itself."""
    src = (ROOT / "build_standalone.py").read_text(encoding="utf-8")
    m = re.search(r'_SHELL = r"""(.*?)\n"""\n', src, re.S)
    assert m, "could not isolate _SHELL"
    assert '"""' not in m.group(1)


# ------------------------------------------------- bundled files are unmodified
@pytest.mark.parametrize("name", ["passes.py", "sites_climatology.json"])
def test_bundled_files_are_byte_identical_to_src(bundle, name):
    """The whole argument for Pyodide over a JS rewrite is that the VALIDATED code
    ships unchanged. If the bundle drifts from src/, that argument is void."""
    assert bundle[name] == (ROOT / "src" / name).read_text(encoding="utf-8")


def test_spa_is_bundled_byte_identical(html):
    m = re.search(r'SPA = "([^"]+)"', html)
    spa = base64.b64decode(m.group(1)).decode("utf-8")
    assert spa == (ROOT / "src" / "index.html").read_text(encoding="utf-8")


def test_sgp4_vendor_set_covers_the_import_graph(bundle):
    """Verified by an isolation test: these 9 modules are exactly the
    reachable graph. wrapper.py MUST be absent so api.py falls back to the pure
    python Satrec; tests.py/wulfgar.py/conveniences.py/exporter.py/omm.py are
    unreachable from passes.py."""
    vendored = {k.split("/", 1)[1] for k in bundle if k.startswith("sgp4/")}
    assert "wrapper.py" not in vendored, "bundling wrapper.py defeats the fallback"
    assert {"api.py", "model.py", "propagation.py", "ext.py", "io.py",
            "earth_gravity.py", "functions.py", "alpha5.py", "__init__.py"} == vendored


# ------------------------------------------- the R7 violation that got it reverted
def test_handler_never_guesses_a_polygon(handler_src):
    """The reverted build did `polygon_name=(names[-1] if len(names)>1 else None)`,
    silently imaging a polygon the user never chose — DEVELOPMENT.md constraint 4 / R7.
    It must let AmbiguousPolygonError propagate and return the name list, exactly
    as app.py does, so the SPA renders its own picker."""
    assert "names[-1]" not in handler_src
    assert "AmbiguousPolygonError" in handler_src
    assert "polygons" in handler_src, "the 422 must carry the polygon list"


def test_handler_validates_the_window(handler_src):
    """The reverted build skipped validate_window (the third front-end to do so):
    days=0 rendered '0 passes' as a normal successful result."""
    assert "validate_window" in handler_src


def test_handler_mirrors_app_error_contract(handler_src):
    """The SPA reads detail.polygons for its picker and shows detail as a message
    otherwise (index.html renderPicker / showErr). Diverging silently breaks it."""
    assert "_env(422" in handler_src
    assert '"detail"' in handler_src


# ------------------------------------------------ capabilities claimed vs wired
def test_cloud_is_actually_wired(handler_src):
    """An earlier build declared cloud 'impossible' because attach_cloud's
    http_get seam is synchronous — while ignoring forecast_json/ensemble_json, the
    same injection point the offline tests use, and having already used the very
    same await-in-JS pattern for TLEs three lines earlier."""
    assert "forecast_json=fc" in handler_src and "ensemble_json=ens" in handler_src
    assert "FORECAST_URL" in handler_src and "ENSEMBLE_URL" in handler_src


def test_tier3_climatology_is_wired(handler_src, bundle):
    assert "sites_climatology.json" in handler_src
    assert "climatology=_CLIM" in handler_src
    clim = json.loads(bundle["sites_climatology.json"])
    assert clim.get("SITEA_100sqkm"), "the bundled climatology must be populated"


def test_manoeuvre_detection_is_wired(handler_src):
    """Dead in the reverted build while it still printed a confident timing sigma —
    the number DEVELOPMENT.md records as blind to a burn. Here it diffs against the
    previous run's TLEs persisted in localStorage."""
    assert "orbit_change" in handler_src
    assert "_manoeuvre_warning" in handler_src


def test_coverage_dependencies_are_loaded(html):
    """shapely and pyproj ship WITH Pyodide (v0.26.2 lockfile), so
    aoi_coverage_fraction runs unchanged — an earlier build claimed otherwise
    without checking."""
    assert '"shapely"' in html and '"pyproj"' in html


def test_xlsx_dependencies_are_installed(html):
    """openpyxl and tzdata are pure-python wheels; micropip can install them, so
    the .xlsx download works. tzdata is required — Pyodide has no system zoneinfo."""
    assert "openpyxl" in html and "tzdata" in html


# --------------------------------------------------------- honesty of the page
def test_page_does_not_claim_capabilities_it_lacks(html):
    """An earlier build shipped a banner asserting cloud/coverage were impossible.
    They are not — guard against that false claim resurfacing anywhere in the page."""
    low = html.lower()
    assert "no cloud" not in low, "cloud works; do not claim otherwise"
    assert "geometry only" not in low


def test_page_is_self_contained_apart_from_named_cdns(html):
    """Everything the page needs is inlined except declared remote hosts. A stray
    relative src= would 404 for a user who just double-clicks the file."""
    assert 'src="./' not in html and "src='./" not in html
    hosts = set(re.findall(r"https://([a-z0-9.-]+)/", html))
    allowed = {"cdn.jsdelivr.net", "celestrak.org", "api.open-meteo.com",
               "ensemble-api.open-meteo.com", "archive-api.open-meteo.com",
               "cdn.plot.ly", "unpkg.com", "fonts.googleapis.com", "fonts.gstatic.com",
               "server.arcgisonline.com", "earth-search.aws.element84.com",
               "basemaps.cartocdn.com", "a.basemaps.cartocdn.com",
               "tile.openstreetmap.org", "www.openstreetmap.org", "carto.com",
               "pypi.org", "files.pythonhosted.org"}
    assert hosts <= allowed, f"unexpected remote host(s): {hosts - allowed}"
