/**
 * Lens bundle schema (`lens_<dataset>.json`, version 1) — typed model + a
 * defensive validator (mirrors lib/sgp.ts / lib/umtvit.ts). A self-contained
 * export for the /lens explorer: class names + taxonomy + the learned
 * malignancy axis + a set of example lesions, each carrying the diagnosis
 * softmax and the CLS feature the three readouts need. In a live workbench these
 * come from the pack (prediction.probabilities + tokens.bin CLS); the bundle
 * inlines them so /lens runs with no backend.
 *
 * The readout MATH lives in lib/malignancy.ts (shared with the workbench path);
 * this module only parses + shapes the standalone bundle.
 */
import type { MalignancyAxis, Taxonomy } from "./malignancy";

export const LENS_SCHEMA_VERSION = 1 as const;

export interface LensLesion {
  id: string;
  thumb_png_b64: string;
  true_label: string | null;
  probabilities: number[];
  feature: number[];
}

export interface LensBundle {
  version: number;
  dataset: string;
  class_names: string[];
  taxonomy: Taxonomy;
  axis: MalignancyAxis | null;
  lesions: LensLesion[];
  provenance: Record<string, unknown>;
}

export class LensValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LensValidationError";
  }
}

function fail(path: string, expected: string): never {
  throw new LensValidationError(`${path}: expected ${expected}`);
}
function obj(v: unknown, path: string): Record<string, unknown> {
  if (v === null || typeof v !== "object" || Array.isArray(v)) fail(path, "an object");
  return v as Record<string, unknown>;
}
function num(v: unknown, path: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) fail(path, "a finite number");
  return v;
}
function str(v: unknown, path: string): string {
  if (typeof v !== "string") fail(path, "a string");
  return v;
}
function arr(v: unknown, path: string): unknown[] {
  if (!Array.isArray(v)) fail(path, "an array");
  return v;
}
function numArray(v: unknown, path: string): number[] {
  return arr(v, path).map((x, i) => num(x, `${path}[${i}]`));
}
function strArray(v: unknown, path: string): string[] {
  return arr(v, path).map((x, i) => str(x, `${path}[${i}]`));
}

function parseTaxonomy(v: unknown, classNames: string[]): Taxonomy {
  const o = obj(v, "taxonomy");
  const mal = obj(o.malignant, "taxonomy.malignant");
  const lvl = obj(o.category_level, "taxonomy.category_level");
  const malignant: Record<string, boolean> = {};
  const category_level: Record<string, number> = {};
  for (const c of classNames) {
    if (typeof mal[c] !== "boolean") fail(`taxonomy.malignant.${c}`, "a boolean");
    malignant[c] = mal[c] as boolean;
    category_level[c] = num(lvl[c], `taxonomy.category_level.${c}`);
  }
  return {
    malignant,
    category_level,
    category_labels: strArray(o.category_labels, "taxonomy.category_labels"),
    axis_pair: Array.isArray(o.axis_pair)
      ? [str(o.axis_pair[0], "taxonomy.axis_pair[0]"), str(o.axis_pair[1], "taxonomy.axis_pair[1]")]
      : undefined,
  };
}

function parseAxis(v: unknown, classDim?: number): MalignancyAxis | null {
  if (v === null || v === undefined) return null;
  const o = obj(v, "axis");
  if (o.provider !== "malignancy_axis") fail("axis.provider", '"malignancy_axis"');
  const dim = num(o.dim, "axis.dim");
  const u = numArray(o.u, "axis.u");
  const cb = numArray(o.centroid_benign, "axis.centroid_benign");
  if (u.length !== dim || cb.length !== dim) fail("axis", `u and centroid_benign of length dim=${dim}`);
  if (classDim !== undefined && classDim !== dim) fail("axis.dim", `to match lesion feature dim ${classDim}`);
  return {
    provider: "malignancy_axis",
    space: str(o.space, "axis.space"),
    dim,
    u,
    centroid_benign: cb,
    anchor_lo: num(o.anchor_lo, "axis.anchor_lo"),
    anchor_hi: num(o.anchor_hi, "axis.anchor_hi"),
    residual_threshold: num(o.residual_threshold, "axis.residual_threshold"),
    provenance: (o.provenance as Record<string, unknown>) ?? {},
  };
}

export function parseLensJson(text: string): LensBundle {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new LensValidationError(`not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
  const o = obj(raw, "$");
  const version = num(o.lens_schema_version, "lens_schema_version");
  if (version !== LENS_SCHEMA_VERSION) {
    throw new LensValidationError(
      `lens_schema_version: ${version} not supported (expected ${LENS_SCHEMA_VERSION})`,
    );
  }
  const class_names = strArray(o.class_names, "class_names");
  const K = class_names.length;
  if (K < 1) fail("class_names", "at least one class");
  const taxonomy = parseTaxonomy(o.taxonomy, class_names);

  const lesions: LensLesion[] = arr(o.lesions, "lesions").map((l, i) => {
    const lo = obj(l, `lesions[${i}]`);
    const probs = numArray(lo.probabilities, `lesions[${i}].probabilities`);
    if (probs.length !== K) fail(`lesions[${i}].probabilities`, `${K} values`);
    const feat = numArray(lo.feature, `lesions[${i}].feature`);
    return {
      id: str(lo.id, `lesions[${i}].id`),
      thumb_png_b64: str(lo.thumb_png_b64, `lesions[${i}].thumb_png_b64`),
      true_label: lo.true_label === null || lo.true_label === undefined ? null : str(lo.true_label, `lesions[${i}].true_label`),
      probabilities: probs,
      feature: feat,
    };
  });
  const featDim = lesions[0]?.feature.length;
  const axis = parseAxis(o.axis, featDim);

  return {
    version,
    dataset: str(o.dataset, "dataset"),
    class_names,
    taxonomy,
    axis,
    lesions,
    provenance: (o.provenance as Record<string, unknown>) ?? {},
  };
}
