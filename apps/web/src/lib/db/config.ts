/**
 * Storage / DB configuration and URL construction (§15). One resolver decides
 * between MOCK mode (bundled fixture under /public/mock, zero backend — how CI
 * and the M6/M7 renderers run) and Supabase mode (anon-key reads + public
 * Storage CDN).
 *
 * MOCK mode is selected when NEXT_PUBLIC_VITREOUS_MOCK=1 OR when either Supabase
 * env var is missing. All env is NEXT_PUBLIC_* (client-inlined); no secrets.
 */

export interface StorageConfig {
  mode: "mock" | "supabase";
  supabaseUrl?: string;
  anonKey?: string;
  /** Public base for the mock fixture. */
  mockBase: string;
  /** Storage bucket holding packs (§15). */
  packBucket: string;
  /**
   * Why {@link StorageConfig.mode} is what it is, in words a deployer can act
   * on. Mock mode is a SILENT fallback — a deploy missing one env var serves
   * the bundled EuroSAT/Pet fixture and looks completely healthy, which is
   * indistinguishable from "my dataset didn't publish" unless the app says so.
   */
  reason: string;
  /** Env vars the deployment is missing (empty when fully configured). */
  missingEnv: string[];
}

export function getStorageConfig(): StorageConfig {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  const forceMock = process.env.NEXT_PUBLIC_VITREOUS_MOCK === "1";

  const missingEnv: string[] = [];
  if (!url) missingEnv.push("NEXT_PUBLIC_SUPABASE_URL");
  if (!key) missingEnv.push("NEXT_PUBLIC_SUPABASE_ANON_KEY");

  const mode: StorageConfig["mode"] = forceMock || missingEnv.length > 0 ? "mock" : "supabase";
  const reason = forceMock
    ? "NEXT_PUBLIC_VITREOUS_MOCK=1 forces the bundled demo fixture."
    : missingEnv.length > 0
      ? `Missing ${missingEnv.join(" and ")} — serving the bundled demo fixture ` +
        "(EuroSAT + Oxford Pet only, no HAM10000). Set these in your Vercel " +
        "project's Environment Variables and redeploy: NEXT_PUBLIC_* values are " +
        "inlined at BUILD time, so adding them without a rebuild changes nothing."
      : "Reading live data from Supabase.";

  return {
    mode,
    supabaseUrl: url,
    anonKey: key,
    mockBase: "/mock",
    packBucket: "packs",
    reason,
    missingEnv,
  };
}

/**
 * Public base URL for a pack directory (WITH trailing slash), built from a
 * `pack_prefix` (already trailing-slashed, §15 handoff). PackClient appends the
 * asset name to this.
 */
export function packUrlFor(cfg: StorageConfig, packPrefix: string): string {
  if (cfg.mode === "mock") {
    return `${cfg.mockBase}/${cfg.packBucket}/${packPrefix}`;
  }
  // Supabase public object URL: {url}/storage/v1/object/public/{bucket}/{key}
  return `${cfg.supabaseUrl}/storage/v1/object/public/${cfg.packBucket}/${packPrefix}`;
}
