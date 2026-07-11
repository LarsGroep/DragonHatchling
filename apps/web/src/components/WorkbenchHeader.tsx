"use client";

/**
 * Minimal light header (UX-VISION-2). Product mark, a couple of quiet readouts
 * (dataset · prediction), the Plain/Expert mode toggle, and a small pack-status
 * lamp. Clean and unobtrusive — the brain is the star, not the chrome.
 */
import Link from "next/link";
import { useWorkbench } from "@/src/lib/state/store";

function ModeToggle() {
  const mode = useWorkbench((s) => s.mode);
  return (
    <div
      className="flex items-center rounded-md border border-edge bg-panel p-0.5"
      role="group"
      aria-label="Explanation mode"
    >
      {(["plain", "expert"] as const).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => useWorkbench.getState().setMode(m)}
          aria-pressed={mode === m}
          title={m === "plain" ? "Plain — lay language, guided" : "Expert — methods, layers, stats"}
          className={`rounded px-2.5 py-1 text-[11px] font-medium capitalize transition-colors ${
            mode === m ? "bg-image/10 text-image" : "text-muted hover:text-readout"
          }`}
        >
          {m}
        </button>
      ))}
    </div>
  );
}

export function WorkbenchHeader({ datasetName }: { datasetName?: string }) {
  const manifest = useWorkbench((s) => s.manifest);
  const loading = useWorkbench((s) => s.loading);
  const error = useWorkbench((s) => s.error);

  const pred = manifest
    ? `${manifest.prediction.label} · ${(manifest.prediction.confidence * 100).toFixed(0)}%`
    : "—";

  const status = error ? "error" : loading ? "loading" : manifest ? "ready" : "no pack";
  const lamp = error
    ? "bg-red-500"
    : loading
      ? "bg-warm animate-pulse"
      : manifest
        ? "bg-evidence"
        : "bg-muted";

  return (
    <header className="flex items-center justify-between border-b border-edge bg-void px-4 py-2.5">
      <div className="flex items-baseline gap-3">
        <span className="text-[15px] font-semibold tracking-tight text-signal">ViTreous</span>
        <span className="hidden text-[12px] text-muted md:inline">
          a window into a vision model&rsquo;s brain
        </span>
        <Link
          href="/umtvit"
          className="rounded-md border border-edge px-2 py-0.5 text-[11px] font-medium text-muted transition-colors hover:border-latent hover:text-latent"
          title="UMT-ViT Explorer — a separate topographic-latent experiment"
        >
          UMT-ViT
        </Link>
      </div>

      <div className="flex items-center gap-4">
        <div className="hidden items-baseline gap-4 sm:flex">
          {datasetName ? (
            <span className="max-w-[22ch] truncate text-[12px] text-muted" title={datasetName}>
              {datasetName}
            </span>
          ) : null}
          <span className="font-mono text-[12px] tabular-nums text-readout" title="prediction">
            {pred}
          </span>
        </div>
        <ModeToggle />
        <div className="flex items-center gap-1.5" title={status}>
          <span className={`h-2 w-2 rounded-full ${lamp}`} />
          <span className="hidden text-[10px] uppercase tracking-wide text-muted lg:inline">
            {status}
          </span>
        </div>
      </div>
    </header>
  );
}
