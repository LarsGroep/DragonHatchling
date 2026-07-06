"use client";

/**
 * Instrument header wired to the store (§1). Product mark plus live monospaced
 * readouts (model / dataset / image / prediction) and a pack-status lamp.
 */
import { useWorkbench } from "@/src/lib/state/store";

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-[10px] uppercase tracking-widest text-muted">{label}</span>
      <span className="max-w-[16ch] truncate tabular-nums text-xs text-readout" title={value}>
        {value}
      </span>
    </div>
  );
}

export function WorkbenchHeader({ datasetName }: { datasetName?: string }) {
  const manifest = useWorkbench((s) => s.manifest);
  const loading = useWorkbench((s) => s.loading);
  const error = useWorkbench((s) => s.error);

  const model = manifest?.model.arch ?? "—";
  const image = manifest?.image.id ?? "—";
  const pred = manifest ? `${manifest.prediction.label} ${(manifest.prediction.confidence * 100).toFixed(0)}%` : "—";

  const status = error ? "error" : loading ? "loading…" : manifest ? "pack loaded" : "no pack";
  const lamp = error ? "bg-gauss" : loading ? "bg-graph animate-pulse" : manifest ? "bg-latent" : "bg-muted";

  return (
    <header className="flex items-center justify-between border-b border-edge bg-panel-hi px-4 py-2.5">
      <div className="flex items-baseline gap-3">
        <span className="text-sm font-semibold tracking-[0.3em] text-signal">VITREOUS</span>
        <span className="hidden text-[11px] text-muted sm:inline">
          explainable vision-transformer workbench
        </span>
      </div>

      <div className="flex items-center gap-5">
        <Readout label="model" value={model} />
        <Readout label="dataset" value={datasetName ?? "—"} />
        <Readout label="image" value={image} />
        <Readout label="pred" value={pred} />
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${lamp}`} />
          <span className="text-[10px] uppercase tracking-widest text-muted">{status}</span>
        </div>
      </div>
    </header>
  );
}
