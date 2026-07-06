"use client";

/**
 * Image Space (§13) — the trust anchor. Original image, 14×14 patch grid,
 * top-5 class-probability bars, dataset/prediction metadata, a switchable
 * per-method saliency overlay with a method-agreement badge, and hover that
 * emits a `patch` EntityRef into the shared store. Highlights reflect the
 * store's hover/pinned selection resolved to patch cells — so hovering here
 * lights up the other views, and their selections light up here.
 */
import { useMemo, useRef, useState } from "react";
import { useWorkbench, layerForT } from "@/src/lib/state/store";
import type { AttributionMethod } from "@/src/lib/state/refs";
import { resolve, resolvedPatches } from "@/src/lib/state/resolver";
import { patchToToken } from "@/src/lib/state/packIndex";
import { useImageSpaceData } from "./useImageSpaceData";
import { SaliencyOverlay } from "./SaliencyOverlay";

const METHODS: AttributionMethod[] = ["chefer", "rollout", "gradcam", "ig"];

/** Mean pairwise agreement of `method` with the others (from faithfulness). */
function agreementScore(
  agreement: Record<string, Record<string, number>> | undefined,
  method: string,
): number | null {
  const row = agreement?.[method];
  if (!row) return null;
  const others = Object.entries(row).filter(([k]) => k !== method);
  if (!others.length) return null;
  return others.reduce((a, [, v]) => a + v, 0) / others.length;
}

export function ImageSpaceView({ classNames }: { classNames?: string[] }) {
  const client = useWorkbench((s) => s.client);
  const manifest = useWorkbench((s) => s.manifest);
  const packIndex = useWorkbench((s) => s.packIndex);
  const t = useWorkbench((s) => s.t);
  const hover = useWorkbench((s) => s.hover);
  const pinned = useWorkbench((s) => s.pinned);
  const method = useWorkbench((s) => s.method);
  const setHover = useWorkbench((s) => s.setHover);
  const togglePin = useWorkbench((s) => s.togglePin);
  const setMethod = useWorkbench((s) => s.setMethod);

  const { attribution, faithfulness } = useImageSpaceData();
  const [showSaliency, setShowSaliency] = useState(true);
  const [opacity, setOpacity] = useState(0.85);
  const stageRef = useRef<HTMLDivElement>(null);

  const grid = packIndex?.grid ?? 14;
  const layer = layerForT(t, packIndex?.numLayers ?? 12);
  const imageUrl = manifest ? client?.imageUrl(manifest) ?? null : null;

  // Concept ids linked to the active selection (§9 tier surfaced in the UI).
  const hoverConcepts = useMemo(() => {
    const active = hover ?? pinned[pinned.length - 1];
    if (!active || !packIndex) return [];
    return resolve(active, packIndex, layerForT(t, packIndex.numLayers)).concepts.slice(0, 6);
  }, [hover, pinned, packIndex, t]);

  // Patch cells lit by the current selection (hover takes precedence, then pins).
  const litPatches = useMemo(() => {
    const set = new Set<string>();
    for (const p of pinned) for (const k of resolvedPatches(p, packIndex, layer)) set.add(k);
    for (const k of resolvedPatches(hover, packIndex, layer)) set.add(k);
    return set;
  }, [hover, pinned, packIndex, layer]);

  const top5 = useMemo(() => {
    const probs = manifest?.prediction.probabilities ?? [];
    return probs
      .map((p, i) => ({ i, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 5);
  }, [manifest]);

  if (!manifest || !packIndex) {
    return (
      <div className="flex h-full items-center justify-center text-[11px] tracking-widest text-muted">
        awaiting pack
      </div>
    );
  }

  const pred = manifest.prediction;
  const agree = agreementScore(faithfulness?.agreement, method);
  const delAuc = faithfulness?.deletion_auc?.[method];

  function handlePointer(e: React.PointerEvent<HTMLDivElement>) {
    const el = stageRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * grid);
    const row = Math.floor(((e.clientY - r.top) / r.height) * grid);
    if (row < 0 || col < 0 || row >= grid || col >= grid) return;
    setHover({ kind: "patch", row, col });
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      {/* method switcher + saliency controls */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          {METHODS.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMethod(m)}
              className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-widest transition-colors ${
                method === m
                  ? "border-image/60 bg-image/10 text-image"
                  : "border-edge bg-panel text-muted hover:text-readout"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-muted">
          <input
            type="checkbox"
            checked={showSaliency}
            onChange={(e) => setShowSaliency(e.target.checked)}
            className="accent-image"
          />
          saliency
        </label>
      </div>

      <div className="flex min-h-0 flex-1 gap-3">
        {/* image stage */}
        <div className="flex min-h-0 flex-1 items-center justify-center">
          <div
            ref={stageRef}
            onPointerMove={handlePointer}
            onPointerLeave={() => setHover(null)}
            onClick={(e) => {
              const el = stageRef.current;
              if (!el) return;
              const r = el.getBoundingClientRect();
              const col = Math.floor(((e.clientX - r.left) / r.width) * grid);
              const row = Math.floor(((e.clientY - r.top) / r.height) * grid);
              if (row >= 0 && col >= 0 && row < grid && col < grid) {
                togglePin({ kind: "token", layer, idx: patchToToken(row, col, grid) });
              }
            }}
            className="relative aspect-square w-full max-w-[min(100%,60vh)] overflow-hidden rounded border border-edge bg-black"
          >
            {imageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={imageUrl}
                alt={pred.label}
                className="absolute inset-0 h-full w-full object-cover"
                style={{ imageRendering: "pixelated" }}
                draggable={false}
              />
            ) : null}

            {showSaliency && attribution ? (
              <SaliencyOverlay
                attribution={attribution}
                grid={grid}
                layer={layer}
                opacity={opacity}
              />
            ) : null}

            {/* patch grid + highlight */}
            <svg
              viewBox={`0 0 ${grid} ${grid}`}
              className="pointer-events-none absolute inset-0 h-full w-full"
              preserveAspectRatio="none"
            >
              {Array.from({ length: grid + 1 }, (_, i) => (
                <g key={i} stroke="#c8d3e6" strokeWidth={0.01} opacity={0.12}>
                  <line x1={i} y1={0} x2={i} y2={grid} />
                  <line x1={0} y1={i} x2={grid} y2={i} />
                </g>
              ))}
              {Array.from(litPatches).map((k) => {
                const [row, col] = k.split(":").map(Number);
                return (
                  <rect
                    key={k}
                    x={col}
                    y={row}
                    width={1}
                    height={1}
                    fill="none"
                    stroke="#e8eefc"
                    strokeWidth={0.08}
                    className="[paint-order:stroke]"
                  />
                );
              })}
            </svg>
          </div>
        </div>

        {/* readout column */}
        <div className="flex w-40 shrink-0 flex-col gap-3 overflow-y-auto">
          <div>
            <div className="text-[10px] uppercase tracking-widest text-muted">prediction</div>
            <div className="truncate text-sm text-signal" title={pred.label}>
              {pred.label}
            </div>
            <div className="tabular-nums text-[11px] text-image">
              {(pred.confidence * 100).toFixed(1)}%
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <div className="text-[10px] uppercase tracking-widest text-muted">top-5</div>
            {top5.map(({ i, p }) => {
              const name = classNames?.[i] ?? `class ${i}`;
              const isPred = i === pred.class_index;
              return (
                <div key={i} className="flex flex-col gap-0.5">
                  <div className="flex items-baseline justify-between gap-1">
                    <span
                      className={`truncate text-[10px] ${isPred ? "text-readout" : "text-muted"}`}
                      title={name}
                    >
                      {name}
                    </span>
                    <span className="tabular-nums text-[9px] text-muted">
                      {(p * 100).toFixed(1)}
                    </span>
                  </div>
                  <div className="h-1 overflow-hidden rounded-full bg-panel-hi">
                    <div
                      className={isPred ? "h-full bg-image" : "h-full bg-muted/50"}
                      style={{ width: `${Math.max(1, p * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* method-agreement badge */}
          <div className="rounded border border-edge bg-panel-hi p-2">
            <div className="text-[10px] uppercase tracking-widest text-muted">
              {method} · faithfulness
            </div>
            {agree != null ? (
              <div className="mt-1 flex items-baseline gap-1">
                <span className="tabular-nums text-sm text-signal">{agree.toFixed(2)}</span>
                <span className="text-[9px] text-muted">mean agreement</span>
              </div>
            ) : (
              <div className="mt-1 text-[10px] text-muted">n/a</div>
            )}
            {delAuc != null ? (
              <div className="text-[9px] text-muted">
                deletion AUC{" "}
                <span className="tabular-nums text-readout">{delAuc.toFixed(3)}</span>
                <span className="text-muted/70"> (lower = better)</span>
              </div>
            ) : null}
          </div>

          {hoverConcepts.length ? (
            <div className="rounded border border-edge bg-panel-hi p-2">
              <div className="text-[10px] uppercase tracking-widest text-muted">concepts (SAE)</div>
              <div className="mt-1 flex flex-wrap gap-1">
                {hoverConcepts.map((c) => (
                  <span key={c} className="tabular-nums rounded bg-panel px-1 text-[9px] text-latent">
                    #{c}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {showSaliency ? (
            <label className="flex flex-col gap-1 text-[10px] uppercase tracking-widest text-muted">
              overlay opacity
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="accent-image"
              />
            </label>
          ) : null}
        </div>
      </div>
    </div>
  );
}
