import { describe, it, expect } from "vitest";
import {
  parseUmtvitBundle,
  parseUmtvitJson,
  UmtvitValidationError,
  UMTVIT_SCHEMA_VERSION,
} from "./umtvit";

/** A minimal but structurally complete valid bundle (2 depth slices, 2×2×2 SOM). */
function validBundle() {
  return {
    version: 1,
    dataset: { name: "shapes", image_size: 64, augmentation: "natural_default", num_classes: 3 },
    model: {
      dim: 96,
      depth: 2,
      volume_grid: 2,
      volume_channels: 24,
      som_grid: [2, 2, 2],
      cross_attention: "cls_bridged",
      params_millions: 0.58,
    },
    metrics: {
      linear_probe: 0.55,
      knn: 0.42,
      chance: 0.333,
      som_quantization_error: 1.68,
      som_topographic_error: 0.21,
      som_dead_fraction: 0.89,
      trustworthiness: null,
    },
    spectral_centroids: [0.22, 0.33],
    classes: ["circle", "square", "triangle"],
    history: {
      steps: [0, 1, 2],
      series: {
        total: [3, 2, 1],
        ntxent: [1, 0.9, 0.8],
        som: [0.5, 0.4, 0.3],
        smooth: [0.1, 0.1, 0.1],
        order: [0.2, 0.2, 0.2],
      },
    },
    probes: [
      {
        label: "circle",
        image_png_b64: "iVBORw0KGgo=",
        cube_initial: [
          [
            [0, 1],
            [0.5, 0.5],
          ],
          [
            [1, 0],
            [0.25, 0.75],
          ],
        ],
        cube_final: [
          [
            [0.1, 0.9],
            [0.5, 0.5],
          ],
          [
            [0.8, 0.2],
            [0.3, 0.7],
          ],
        ],
      },
    ],
    som: {
      grid: [2, 2, 2],
      epochs: [0, 1],
      umatrix: [
        [
          [
            [0.1, 0.2],
            [0.3, 0.4],
          ],
          [
            [0.5, 0.6],
            [0.7, 0.8],
          ],
        ],
        [
          [
            [0.2, 0.3],
            [0.4, 0.5],
          ],
          [
            [0.6, 0.7],
            [0.8, 0.9],
          ],
        ],
      ],
      hits_final: [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [0, 5],
          [6, 7],
        ],
      ],
    },
    embeddings: {
      epochs: [0, 1],
      coords: [
        [
          [0.1, 0.2],
          [0.3, 0.4],
        ],
        [
          [0.5, 0.6],
          [0.7, 0.8],
        ],
      ],
      labels: [0, 1],
    },
  };
}

describe("parseUmtvitBundle", () => {
  it("accepts a valid fixture and narrows the type", () => {
    const b = parseUmtvitBundle(validBundle());
    expect(b.version).toBe(UMTVIT_SCHEMA_VERSION);
    expect(b.dataset.name).toBe("shapes");
    expect(b.model.som_grid).toEqual([2, 2, 2]);
    expect(b.metrics.trustworthiness).toBeNull();
    expect(b.probes[0].cube_final.length).toBe(2);
    expect(b.som.umatrix.length).toBe(2);
    expect(b.embeddings.coords[0].length).toBe(2);
  });

  it("accepts unlabeled bundles (classes null, labels null)", () => {
    const raw = validBundle();
    raw.classes = null as unknown as string[];
    raw.embeddings.labels = null as unknown as number[];
    raw.dataset.num_classes = 0;
    const b = parseUmtvitBundle(raw);
    expect(b.classes).toBeNull();
    expect(b.embeddings.labels).toBeNull();
  });

  it("accepts empty history series (all data unlabeled/short runs)", () => {
    const raw = validBundle();
    raw.history.series.total = [];
    const b = parseUmtvitBundle(raw);
    expect(b.history.series.total).toEqual([]);
  });

  it("rejects the wrong schema version by name", () => {
    const raw = validBundle();
    raw.version = 2;
    expect(() => parseUmtvitBundle(raw)).toThrow(UmtvitValidationError);
    expect(() => parseUmtvitBundle(raw)).toThrow(/version: unsupported bundle version 2/);
  });

  it("rejects a missing top-level field, naming it", () => {
    const raw = validBundle() as Record<string, unknown>;
    delete raw.metrics;
    expect(() => parseUmtvitBundle(raw)).toThrow(/metrics: expected an object/);
  });

  it("rejects a missing nested field, naming its path", () => {
    const raw = validBundle();
    delete (raw.model as Record<string, unknown>).volume_grid;
    expect(() => parseUmtvitBundle(raw)).toThrow(/model\.volume_grid: expected a finite number/);
  });

  it("rejects a wrong-length som_grid", () => {
    const raw = validBundle();
    raw.model.som_grid = [4, 4];
    expect(() => parseUmtvitBundle(raw)).toThrow(/model\.som_grid: expected 3 values/);
  });

  it("rejects a ragged latent cube, naming the slice", () => {
    const raw = validBundle();
    // second slice's second row has width 3 instead of 2
    raw.probes[0].cube_final[1][1] = [0.3, 0.7, 0.9];
    expect(() => parseUmtvitBundle(raw)).toThrow(/probes\[0\]\.cube_final\[1\]\[1\]: ragged array/);
  });

  it("rejects a ragged U-matrix", () => {
    const raw = validBundle();
    raw.som.umatrix[0][0] = [[0.1, 0.2, 0.3]];
    expect(() => parseUmtvitBundle(raw)).toThrow(/som\.umatrix\[0\].*ragged array/);
  });

  it("rejects a non-2D embedding point", () => {
    const raw = validBundle();
    raw.embeddings.coords[0] = [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ];
    expect(() => parseUmtvitBundle(raw)).toThrow(/embeddings\.coords\[0\]/);
  });

  it("rejects a history series whose length disagrees with steps", () => {
    const raw = validBundle();
    raw.history.series.som = [0.5, 0.4];
    expect(() => parseUmtvitBundle(raw)).toThrow(/history\.series\.som: length 2 does not match/);
  });

  it("rejects a NaN metric (JSON-injected) but accepts null", () => {
    const raw = validBundle() as unknown as { metrics: Record<string, unknown> };
    raw.metrics.knn = Number.NaN;
    expect(() => parseUmtvitBundle(raw)).toThrow(/metrics\.knn: expected a finite number or null/);
  });

  it("rejects a non-object bundle", () => {
    expect(() => parseUmtvitBundle(null)).toThrow(/bundle: expected an object/);
    expect(() => parseUmtvitBundle([1, 2, 3])).toThrow(/bundle: expected an object/);
  });
});

describe("parseUmtvitJson", () => {
  it("round-trips a serialized valid bundle", () => {
    const b = parseUmtvitJson(JSON.stringify(validBundle()));
    expect(b.dataset.name).toBe("shapes");
  });

  it("wraps JSON syntax errors as validation errors", () => {
    expect(() => parseUmtvitJson("{not json")).toThrow(UmtvitValidationError);
    expect(() => parseUmtvitJson("{not json")).toThrow(/not valid JSON/);
  });
});
