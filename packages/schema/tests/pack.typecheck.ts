/**
 * TypeScript arm of the pack manifest round-trip test.
 *
 * Imports the shared fixture JSON and checks it against `PackManifest` at
 * compile time. `resolveJsonModule` widens JSON string literals (e.g.
 * `image.source` becomes `string`, not `"gallery"`), so the string-literal
 * union fields are narrowed explicitly; every other field is checked
 * structurally by the `PackManifest` annotation, so shape/presence/number-type
 * drift breaks `tsc --noEmit`. The Python/jsonschema arms live in
 * packages/core/tests.
 */
import fixture from "../fixtures/manifest.fixture.json";
import type {
  AssetIndex,
  ImageSource,
  PackManifest,
} from "../src/pack";

// The load-bearing check: the annotation forces structural conformance.
// Only the JSON-widened string-literal unions are re-narrowed via `as`.
const manifest: PackManifest = {
  ...fixture,
  image: {
    ...fixture.image,
    source: fixture.image.source as ImageSource,
  },
  assets: fixture.assets as AssetIndex,
};

// Field-level probes so the check is meaningful, not just import-level.
const _label: string = manifest.prediction.label;
const _conf: number = manifest.prediction.confidence;
const _source: ImageSource = manifest.image.source;
const _firstAsset: string = Object.keys(manifest.assets)[0];

void _label;
void _conf;
void _source;
void _firstAsset;

export {};
