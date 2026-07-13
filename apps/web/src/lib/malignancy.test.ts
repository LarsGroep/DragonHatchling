/**
 * Malignancy lens web mirror — parity with the Python core (vitreous.malignancy)
 * on the same synthetic setup, headless.
 */
import { describe, expect, it } from "vitest";
import type { LoadedTokens } from "./pack/types";
import {
  categoryLevels,
  clsFinalFeature,
  expectedCategory,
  hardCategory,
  malignantIndices,
  malignantProbability,
  projectFeature,
  type MalignancyAxis,
  type Taxonomy,
} from "./malignancy";

const CLASSES = ["nevus", "keratosis", "insitu", "melanoma"];
const TAX: Taxonomy = {
  malignant: { nevus: false, keratosis: false, insitu: true, melanoma: true },
  category_level: { nevus: 0, keratosis: 0, insitu: 1, melanoma: 2 },
  category_labels: ["benign", "in-situ", "invasive"],
};

describe("derived readouts", () => {
  it("sums malignant probability over the right classes", () => {
    const idx = malignantIndices(CLASSES, TAX);
    expect(idx).toEqual([2, 3]);
    expect(malignantProbability([0.6, 0.1, 0.1, 0.2], idx)).toBeCloseTo(0.3);
    expect(malignantProbability([0.6, 0.1, 0.1, 0.2], [])).toBe(0);
  });

  it("computes the expected + hard category", () => {
    const levels = categoryLevels(CLASSES, TAX);
    expect(levels).toEqual([0, 0, 1, 2]);
    expect(expectedCategory([0, 0, 0, 1], levels)).toBeCloseTo(2);
    expect(expectedCategory([0.5, 0, 0, 0.5], levels)).toBeCloseTo(1);
    expect(expectedCategory([0.1, 0.1, 0.8, 0], levels)).toBeCloseTo(0.8);
    expect(hardCategory([0.6, 0, 0, 0.4], levels)).toBe(0);
    expect(hardCategory([0, 0, 0, 1], levels)).toBe(2);
  });

  it("handles zero probability mass", () => {
    expect(expectedCategory([0, 0, 0, 0], [0, 0, 1, 2])).toBe(0);
  });
});

describe("clsFinalFeature", () => {
  it("extracts the CLS row of the final timeline step", () => {
    // steps=2, tokens=3, dim=2. CLS(0) at final step = values [10,11].
    const data = new Float32Array([
      // step 0
      0, 0, 1, 1, 2, 2,
      // step 1  (CLS row first)
      10, 11, 20, 21, 30, 31,
    ]);
    const tokens: LoadedTokens = { data, steps: 2, tokens: 3, dim: 2 };
    expect(Array.from(clsFinalFeature(tokens))).toEqual([10, 11]);
  });
});

describe("projectFeature (parity with the Python axis math)", () => {
  // axis along dim 0, benign centroid at x=-3, calibrated [-3, +3].
  const axis: MalignancyAxis = {
    provider: "malignancy_axis",
    space: "cls_final",
    dim: 3,
    u: [1, 0, 0],
    centroid_benign: [-3, 0, 0],
    anchor_lo: -3,
    anchor_hi: 3,
    residual_threshold: 2.0,
  };

  it("orders benign below malignant, mid in the middle", () => {
    expect(projectFeature([-3, 0, 0], axis).position).toBeCloseTo(0);
    expect(projectFeature([3, 0, 0], axis).position).toBeCloseTo(1);
    expect(projectFeature([0, 0, 0], axis).position).toBeCloseTo(0.5);
  });

  it("clamps beyond the anchors", () => {
    expect(projectFeature([-10, 0, 0], axis).position).toBe(0);
    expect(projectFeature([10, 0, 0], axis).position).toBe(1);
  });

  it("flags OOD when the residual off the axis exceeds threshold", () => {
    expect(projectFeature([0, 0.5, 0], axis).ood).toBe(false);
    const off = projectFeature([0, 5, 0], axis);
    expect(off.residual).toBeCloseTo(5);
    expect(off.ood).toBe(true);
  });
});
