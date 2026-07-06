"use client";

/**
 * A placeholder pane for a view not yet built (Gaussian field M6, graph /
 * embeddings M7). It is NOT inert: it subscribes to the SAME selection store
 * and renders the resolver's output for the current hover/pin, so the four-way
 * synchronization is visible and testable before the real WebGL renderers land
 * — hovering a patch in Image Space lights up a token/node/gaussian readout
 * here. This is the debug surface §11 calls for.
 */
import { useMemo } from "react";
import { useWorkbench, layerForT } from "@/src/lib/state/store";
import { resolve } from "@/src/lib/state/resolver";
import type { PanelAccent } from "@/components/WorkbenchPanel";

const ACCENT_TEXT: Record<PanelAccent, string> = {
  image: "text-image",
  gauss: "text-gauss",
  graph: "text-graph",
  latent: "text-latent",
};

export function SyncedPlaceholder({
  accent,
  hint,
  emphasis,
}: {
  accent: PanelAccent;
  hint: string;
  /** Which resolved field this view is "responsible" for (highlighted). */
  emphasis: "gaussian" | "node" | "point";
}) {
  const packIndex = useWorkbench((s) => s.packIndex);
  const t = useWorkbench((s) => s.t);
  const hover = useWorkbench((s) => s.hover);
  const pinned = useWorkbench((s) => s.pinned);

  const layer = layerForT(t, packIndex?.numLayers ?? 12);
  const active = hover ?? pinned[pinned.length - 1] ?? null;

  const resolved = useMemo(() => {
    if (!active || !packIndex) return null;
    return resolve(active, packIndex, layer);
  }, [active, packIndex, layer]);

  const line = useMemo(() => {
    if (!resolved) return null;
    if (emphasis === "gaussian" && resolved.gaussian)
      return `gaussian L${resolved.gaussian.layer} · #${resolved.gaussian.idx}`;
    if (emphasis === "node" && resolved.node) return resolved.node;
    if (emphasis === "point" && resolved.point)
      return `CLS trajectory · L${resolved.point.layer}`;
    if (emphasis === "point")
      return resolved.isCls ? "CLS" : `token #${resolved.idx} (no point)`;
    return null;
  }, [resolved, emphasis]);

  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
      {resolved && line ? (
        <>
          <span className={`text-sm tabular-nums ${ACCENT_TEXT[accent]}`}>{line}</span>
          <span className="text-[10px] tracking-widest text-muted">
            {resolved.concepts.length
              ? `${resolved.concepts.length} concept${resolved.concepts.length > 1 ? "s" : ""} · `
              : ""}
            {resolved.refs.length} linked refs · synced
          </span>
          <span className="max-w-[30ch] text-[10px] leading-relaxed text-muted/60">{hint}</span>
        </>
      ) : (
        <>
          <span className="text-[11px] tracking-widest text-muted">awaiting selection</span>
          <span className="max-w-[30ch] text-[11px] leading-relaxed text-muted/70">{hint}</span>
        </>
      )}
    </div>
  );
}
