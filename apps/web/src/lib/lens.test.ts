/** Lens bundle parser — happy path + field-named rejections (headless). */
import { describe, expect, it } from "vitest";
import { LensValidationError, parseLensJson } from "./lens";

function valid(): Record<string, unknown> {
  return {
    lens_schema_version: 1,
    dataset: "ham10000",
    class_names: ["nevus", "insitu", "melanoma"],
    taxonomy: {
      malignant: { nevus: false, insitu: true, melanoma: true },
      category_level: { nevus: 0, insitu: 1, melanoma: 2 },
      category_labels: ["benign", "in-situ", "invasive"],
      axis_pair: ["benign", "malignant"],
    },
    axis: {
      provider: "malignancy_axis",
      space: "cls_final",
      dim: 2,
      u: [1, 0],
      centroid_benign: [-1, 0],
      anchor_lo: -1,
      anchor_hi: 1,
      residual_threshold: 2,
    },
    lesions: [
      { id: "a", thumb_png_b64: "x", true_label: "melanoma", probabilities: [0.1, 0.2, 0.7], feature: [1, 0] },
      { id: "b", thumb_png_b64: "y", true_label: null, probabilities: [0.8, 0.1, 0.1], feature: [-1, 0] },
    ],
    provenance: {},
  };
}

describe("parseLensJson", () => {
  it("accepts a valid bundle", () => {
    const b = parseLensJson(JSON.stringify(valid()));
    expect(b.class_names).toHaveLength(3);
    expect(b.taxonomy.malignant.melanoma).toBe(true);
    expect(b.axis?.dim).toBe(2);
    expect(b.lesions[1].true_label).toBeNull();
  });

  it("tolerates a missing axis (null)", () => {
    const doc = valid();
    doc.axis = null;
    expect(parseLensJson(JSON.stringify(doc)).axis).toBeNull();
  });

  it("rejects non-JSON, wrong version, and names them", () => {
    expect(() => parseLensJson("{oops")).toThrow(/not valid JSON/);
    const doc = valid();
    doc.lens_schema_version = 2;
    expect(() => parseLensJson(JSON.stringify(doc))).toThrow(/lens_schema_version/);
  });

  it("rejects a taxonomy missing a class, by name", () => {
    const doc = valid();
    delete (doc.taxonomy as { malignant: Record<string, unknown> }).malignant.melanoma;
    expect(() => parseLensJson(JSON.stringify(doc))).toThrow(/taxonomy\.malignant\.melanoma/);
  });

  it("rejects a probabilities vector of the wrong width", () => {
    const doc = valid();
    (doc.lesions as Array<{ probabilities: number[] }>)[0].probabilities = [0.5, 0.5];
    expect(() => parseLensJson(JSON.stringify(doc))).toThrow(/lesions\[0\]\.probabilities/);
  });

  it("rejects an axis whose dim disagrees with the feature dim", () => {
    const doc = valid();
    (doc.axis as { dim: number; u: number[]; centroid_benign: number[] }).dim = 3;
    (doc.axis as { u: number[] }).u = [1, 0, 0];
    (doc.axis as { centroid_benign: number[] }).centroid_benign = [-1, 0, 0];
    expect(() => parseLensJson(JSON.stringify(doc))).toThrow(/axis\.dim/);
  });

  it("throws LensValidationError instances", () => {
    expect(() => parseLensJson("{oops")).toThrow(LensValidationError);
  });
});

describe("synthetic-bundle detection", () => {
  it("does not flag a plain bundle with no synthetic markers", () => {
    expect(parseLensJson(JSON.stringify(valid())).synthetic).toBe(false);
  });

  it("honours an explicit provenance.synthetic flag", () => {
    const doc = valid();
    doc.provenance = { synthetic: true };
    expect(parseLensJson(JSON.stringify(doc)).synthetic).toBe(true);
  });

  it("recognises a fixture generated before the explicit flag existed", () => {
    // Older demo.json carried only the generator name and the axis note.
    const byGenerator = valid();
    byGenerator.provenance = { generator: "apps/web/scripts/gen-lens-demo.py", seed: 7 };
    expect(parseLensJson(JSON.stringify(byGenerator)).synthetic).toBe(true);

    const byAxisNote = valid();
    (byAxisNote.axis as Record<string, unknown>).provenance = { note: "synthetic axis" };
    expect(parseLensJson(JSON.stringify(byAxisNote)).synthetic).toBe(true);

    const byDatasetSuffix = valid();
    byDatasetSuffix.dataset = "ham10000-demo";
    expect(parseLensJson(JSON.stringify(byDatasetSuffix)).synthetic).toBe(true);
  });

  it("does not flag a real export that merely carries provenance", () => {
    // A genuine run records how its probabilities were obtained; that is an
    // honesty marker about model quality, not a claim the numbers are invented.
    const doc = valid();
    doc.provenance = {
      generator: "kaggle_umtvit_sgp.ipynb §8b",
      probabilities: "SSL-probe ESTIMATE, not a diagnostic softmax",
    };
    expect(parseLensJson(JSON.stringify(doc)).synthetic).toBe(false);
  });
});
