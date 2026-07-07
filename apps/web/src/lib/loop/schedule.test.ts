import { describe, expect, it } from "vitest";
import {
  LOOP_BASE_LPS,
  LOOP_EASE_MIN,
  LOOP_STAGES,
  TIMELINE_MAX,
  clamp01,
  easeInOut,
  loopVelocity,
  stageCopy,
  stageForT,
  verdictProgress,
} from "./schedule";

describe("easeInOut", () => {
  it("pins the endpoints and midpoint", () => {
    expect(easeInOut(0)).toBe(0);
    expect(easeInOut(1)).toBe(1);
    expect(easeInOut(0.5)).toBeCloseTo(0.5, 6);
  });
  it("clamps out-of-range inputs", () => {
    expect(easeInOut(-3)).toBe(0);
    expect(easeInOut(9)).toBe(1);
  });
  it("is monotonically increasing", () => {
    let prev = -1;
    for (let x = 0; x <= 1.0001; x += 0.05) {
      const y = easeInOut(x);
      expect(y).toBeGreaterThanOrEqual(prev);
      prev = y;
    }
  });
});

describe("clamp01", () => {
  it("clamps both ends", () => {
    expect(clamp01(-1)).toBe(0);
    expect(clamp01(2)).toBe(1);
    expect(clamp01(0.4)).toBe(0.4);
  });
});

describe("stageForT", () => {
  it("maps t to the right stage across all boundaries", () => {
    expect(LOOP_STAGES[stageForT(0)].id).toBe("patchify");
    expect(LOOP_STAGES[stageForT(0.9)].id).toBe("patchify");
    expect(LOOP_STAGES[stageForT(1)].id).toBe("attention");
    expect(LOOP_STAGES[stageForT(4.9)].id).toBe("attention");
    expect(LOOP_STAGES[stageForT(5)].id).toBe("strengthen");
    expect(LOOP_STAGES[stageForT(8)].id).toBe("concentrate");
    expect(LOOP_STAGES[stageForT(11)].id).toBe("verdict");
  });
  it("clamps t == TIMELINE_MAX to the final (verdict) stage", () => {
    expect(LOOP_STAGES[stageForT(TIMELINE_MAX)].id).toBe("verdict");
    expect(LOOP_STAGES[stageForT(99)].id).toBe("verdict");
  });
  it("covers the whole timeline with no gaps", () => {
    for (let t = 0; t <= TIMELINE_MAX; t += 0.25) {
      const i = stageForT(t);
      expect(i).toBeGreaterThanOrEqual(0);
      expect(i).toBeLessThan(LOOP_STAGES.length);
    }
  });
});

describe("stageCopy", () => {
  it("returns distinct plain vs expert captions", () => {
    const s = LOOP_STAGES[1];
    expect(stageCopy(s, "plain").caption).not.toBe(stageCopy(s, "expert").caption);
    expect(stageCopy(s, "expert").caption).toMatch(/layers?/i);
  });
});

describe("loopVelocity", () => {
  it("is always positive on the timeline", () => {
    for (let t = 0; t <= TIMELINE_MAX; t += 0.1) {
      expect(loopVelocity(t)).toBeGreaterThan(0);
    }
  });
  it("dips to the ease floor at a stage boundary and peaks mid-stage", () => {
    // attention stage spans t 1..5 → midpoint t=3.
    const boundary = loopVelocity(1);
    const midpoint = loopVelocity(3);
    expect(boundary).toBeCloseTo(LOOP_BASE_LPS * LOOP_EASE_MIN, 5);
    expect(midpoint).toBeCloseTo(LOOP_BASE_LPS, 5);
    expect(midpoint).toBeGreaterThan(boundary);
  });
});

describe("verdictProgress", () => {
  it("is 0 before the verdict stage and 1 at the end", () => {
    expect(verdictProgress(5)).toBe(0);
    expect(verdictProgress(11)).toBe(0);
    expect(verdictProgress(TIMELINE_MAX)).toBe(1);
  });
  it("rises monotonically across the verdict stage", () => {
    expect(verdictProgress(11.5)).toBeGreaterThan(0);
    expect(verdictProgress(11.9)).toBeGreaterThan(verdictProgress(11.5));
  });
});
