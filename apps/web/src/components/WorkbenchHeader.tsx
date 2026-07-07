"use client";

/**
 * Instrument header wired to the store (§1). Product mark, live monospaced
 * readouts (model / dataset / image / prediction), a pack-status lamp, and the
 * Plain/Expert mode toggle (S1). In Plain mode the jargon readouts carry
 * lay-language tooltips (title attributes) — the dark-instrument look is
 * unchanged; only the words soften.
 */
import { useWorkbench } from "@/src/lib/state/store";

function Readout({
  label,
  value,
  plainHint,
  plain,
}: {
  label: string;
  value: string;
  plainHint?: string;
  plain: boolean;
}) {
  const title = plain && plainHint ? plainHint : value;
  return (
    <div className="flex items-baseline gap-2" title={title}>
      <span className="text-[10px] uppercase tracking-widest text-muted">{label}</span>
      <span className="max-w-[16ch] truncate tabular-nums text-xs text-readout">{value}</span>
    </div>
  );
}

function ModeToggle() {
  const mode = useWorkbench((s) => s.mode);
  return (
    <div
      className="flex items-center rounded border border-edge bg-panel p-0.5"
      role="group"
      aria-label="Explanation mode"
    >
      {(["plain", "expert"] as const).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => useWorkbench.getState().setMode(m)}
          aria-pressed={mode === m}
          title={
            m === "plain"
              ? "Plain — lay language, guided"
              : "Expert — methods, layers, AUCs"
          }
          className={`rounded px-2 py-0.5 text-[10px] uppercase tracking-widest transition-colors ${
            mode === m
              ? "bg-signal/10 text-signal"
              : "text-muted hover:text-readout"
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
  const mode = useWorkbench((s) => s.mode);
  const plain = mode === "plain";

  const model = manifest?.model.arch ?? "—";
  const image = manifest?.image.id ?? "—";
  const pred = manifest
    ? `${manifest.prediction.label} ${(manifest.prediction.confidence * 100).toFixed(0)}%`
    : "—";

  const status = error ? "error" : loading ? "loading…" : manifest ? "pack loaded" : "no pack";
  const lamp = error
    ? "bg-gauss"
    : loading
      ? "bg-graph animate-pulse"
      : manifest
        ? "bg-latent"
        : "bg-muted";

  return (
    <header className="flex items-center justify-between border-b border-edge bg-panel-hi px-4 py-2.5">
      <div className="flex items-baseline gap-3">
        <span className="text-sm font-semibold tracking-[0.3em] text-signal">VITREOUS</span>
        <span className="hidden text-[11px] text-muted sm:inline">
          {plain ? "watch a vision model think" : "explainable vision-transformer workbench"}
        </span>
      </div>

      <div className="flex items-center gap-5">
        <Readout
          label="model"
          value={model}
          plain={plain}
          plainHint="The AI being examined — a vision transformer that reads images."
        />
        <Readout label="dataset" value={datasetName ?? "—"} plain={plain} plainHint="The kind of photos it was trained on." />
        <Readout label="image" value={image} plain={plain} plainHint="The photo it is looking at right now." />
        <Readout
          label="pred"
          value={pred}
          plain={plain}
          plainHint="Its answer, and how sure it is (0–100%)."
        />
        <ModeToggle />
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${lamp}`} />
          <span className="text-[10px] uppercase tracking-widest text-muted">{status}</span>
        </div>
      </div>
    </header>
  );
}
