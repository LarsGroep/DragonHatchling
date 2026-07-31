/**
 * Surface flags. The product is ViTreous (`/` + `/lens`); the UMT-ViT
 * experiment line (`/umtvit`, `/sgp`) is hidden unless explicitly enabled.
 *
 * Pinned because the failure is silent in both directions: a stray default
 * would ship the experiment surfaces to users, and a nav link left behind
 * would point at a 404.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { experimentsEnabled } from "./features";

const ORIGINAL = process.env.NEXT_PUBLIC_VITREOUS_EXPERIMENTS;

afterEach(() => {
  if (ORIGINAL === undefined) delete process.env.NEXT_PUBLIC_VITREOUS_EXPERIMENTS;
  else process.env.NEXT_PUBLIC_VITREOUS_EXPERIMENTS = ORIGINAL;
});

function src(rel: string): string {
  return readFileSync(fileURLToPath(new URL(rel, import.meta.url)), "utf-8");
}

describe("experimentsEnabled", () => {
  it("is off unless explicitly set to 1", () => {
    delete process.env.NEXT_PUBLIC_VITREOUS_EXPERIMENTS;
    expect(experimentsEnabled()).toBe(false);

    for (const v of ["", "0", "false", "true", "yes"]) {
      process.env.NEXT_PUBLIC_VITREOUS_EXPERIMENTS = v;
      expect(experimentsEnabled(), `value ${JSON.stringify(v)}`).toBe(false);
    }
  });

  it("is on for exactly \"1\"", () => {
    process.env.NEXT_PUBLIC_VITREOUS_EXPERIMENTS = "1";
    expect(experimentsEnabled()).toBe(true);
  });
});

describe("hidden routes are actually gated", () => {
  it("both experiment pages call notFound() behind the flag", () => {
    for (const page of ["../../app/sgp/page.tsx", "../../app/umtvit/page.tsx"]) {
      const text = src(page);
      expect(text, page).toContain("experimentsEnabled()");
      expect(text, page).toContain("notFound()");
    }
  });

  it("no always-visible link points at a hidden route", () => {
    // Every /umtvit or /sgp href must sit behind an experimentsEnabled() guard.
    for (const file of [
      "../components/WorkbenchHeader.tsx",
      "../components/lens/LensExplorer.tsx",
    ]) {
      const text = src(file);
      const linksToHidden = /href="\/(umtvit|sgp)"/.test(text);
      if (linksToHidden) {
        expect(text, file).toContain("experimentsEnabled()");
      }
    }
  });
});
