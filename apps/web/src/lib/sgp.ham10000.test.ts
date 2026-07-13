/**
 * Contract round-trip for the SHIPPED real run: public/sgp/ham10000.json is
 * SGP run 3's Kaggle export (docs/SGP-RUNS.md) and the /sgp page's default
 * bundle — it must always parse through the strict validator with the
 * geometry and health the run log records.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { depthActivations, migrationCurve, parseSgpJson } from "./sgp";

const FIXTURE = fileURLToPath(new URL("../../public/sgp/ham10000.json", import.meta.url));

describe("shipped HAM10000 bundle (SGP run 3) round-trip", () => {
  const bundle = parseSgpJson(readFileSync(FIXTURE, "utf-8"));

  it("is the run-3 geometry: 6×6×6, depth 8, 16×16 volume, 8 probes", () => {
    expect(bundle.dataset).toBe("ham10000");
    expect(bundle.som.grid).toEqual([6, 6, 6]);
    expect(bundle.som.num_neurons).toBe(216);
    // 6-connected 6³ lattice: 3 · 5 · 6 · 6 face edges.
    expect(bundle.som.edges).toHaveLength(3 * 5 * 6 * 6);
    expect(bundle.som.depth_steps).toBe(8);
    expect(bundle.som.volume_grid).toEqual([16, 16]);
    expect(bundle.probes).toHaveLength(8);
  });

  it("is the healthy map the run log records (207/216 used)", () => {
    const used = bundle.som.nodes.filter((n) => n.hits > 0).length;
    expect(used).toBe(207);
    expect(bundle.som.dead_neurons).toBe(9);
    for (const n of bundle.som.nodes) expect(n.dead).toBe(n.hits === 0);
  });

  it("carries the run-3 eval metrics in provenance (shown on the page)", () => {
    const ev = bundle.som.provenance.eval as Record<string, number>;
    expect(ev.linear_probe).toBeGreaterThan(0.79);
    expect(ev.som_topographic_error).toBeLessThan(0.03);
    expect(ev.chance).toBeCloseTo(1 / 7, 5);
  });

  it("derives well-formed activations + migration for every probe", () => {
    for (const p of bundle.probes) {
      const acts = depthActivations(p.bmu, bundle.som.num_neurons);
      for (const a of acts) {
        let s = 0;
        for (const v of a) s += v;
        expect(s).toBeCloseTo(1, 5);
      }
      for (const m of migrationCurve(p.bmu)) {
        expect(m).toBeGreaterThanOrEqual(0);
        expect(m).toBeLessThanOrEqual(1);
      }
    }
  });
});
