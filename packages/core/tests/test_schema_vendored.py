"""The vendored pack schema must ship with the package and match the source.

build_pack() validates every manifest against the schema at runtime; when
packages/core is pip-installed standalone (Kaggle) the packages/schema source
is absent, so a copy is vendored next to vitreous.packs and declared as
package-data. This guards against (a) the vendored copy going missing and
(b) drift from the single source of truth.
"""
import json
from pathlib import Path

from vitreous.packs import load_pack_schema, SCHEMA_PATH


def test_vendored_schema_exists_next_to_package():
    vendored = Path(__file__).resolve().parents[1] / "src" / "vitreous" / "packs" / "pack.schema.json"
    assert vendored.is_file(), "vendored pack.schema.json missing — wheel would break on Kaggle"


def test_load_pack_schema_prefers_vendored():
    assert Path(SCHEMA_PATH).name == "pack.schema.json"
    assert "packs" in str(SCHEMA_PATH)  # the vendored copy, not packages/schema
    schema = load_pack_schema()
    assert schema.get("$schema", "").startswith("https://json-schema.org/")


def test_vendored_matches_source_of_truth():
    src = Path(__file__).resolve().parents[2] / "schema" / "schema" / "pack.schema.json"
    if not src.is_file():
        return  # not a monorepo checkout; nothing to compare against
    vendored = Path(__file__).resolve().parents[1] / "src" / "vitreous" / "packs" / "pack.schema.json"
    assert json.loads(vendored.read_text()) == json.loads(src.read_text()), (
        "vendored pack.schema.json drifted from packages/schema — re-copy it"
    )
