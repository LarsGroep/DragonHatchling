"use client";

/**
 * Interaction Graph (§8) — canvas-2D renderer, two modes behind a toggle:
 *  • LAYER: the current layer's 197 tokens anchored to their spatial grid
 *    (so the graph stays legible as an image), top-k attention edges with
 *    weight-scaled luminance, Louvain community hues, CLS in a gutter.
 *  • UNROLLED: all 12 layers as vertical strata (x=layer, y=token), that
 *    layer's edges drawn within each stratum; residual (t,i)→(t+1,i) edges
 *    are implicit per graph.json's `residual` flag and drawn only for the
 *    hovered/pinned token to keep 2.4k nodes legible.
 * Hover emits {kind:"node"} EntityRefs; store selections highlight here.
 * Canvas-2D (not WebGL) is deliberate: ≤2.4k dots + ≤1.6k lines per frame is
 * comfortably 60fps and keeps the view dependency-free.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphJson } from "@/src/lib/pack/types";
import { useWorkbench, layerForT } from "@/src/lib/state/store";
import { resolve } from "@/src/lib/state/resolver";
import { nodeId, tokenToPatch } from "@/src/lib/state/packIndex";

// Soft, desaturated community hues for the light theme (no neon).
const HUES = [210, 258, 160, 32, 190, 340, 96, 280, 130, 300, 18, 174];

function nodeXY(idx: number, grid: number, w: number, h: number): [number, number] {
  const pc = tokenToPatch(idx, grid);
  if (!pc) return [w * 0.045, h * 0.06]; // CLS gutter
  const pad = 0.09;
  return [
    (pad + ((pc[1] + 0.5) / grid) * (1 - 2 * pad)) * w,
    (pad + ((pc[0] + 0.5) / grid) * (1 - 2 * pad)) * h,
  ];
}

export function GraphView() {
  const client = useWorkbench((s) => s.client);
  const manifest = useWorkbench((s) => s.manifest);
  const packIndex = useWorkbench((s) => s.packIndex);
  const setHover = useWorkbench((s) => s.setHover);
  const togglePin = useWorkbench((s) => s.togglePin);
  const [graph, setGraph] = useState<GraphJson | null>(null);
  const [mode, setMode] = useState<"layer" | "unrolled">("layer");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let alive = true;
    setGraph(null);
    if (!client || !manifest || !("graph.json" in manifest.assets)) return;
    client.loadGraph().then((g) => alive && setGraph(g)).catch(() => alive && setGraph(null));
    return () => {
      alive = false;
    };
  }, [client, manifest]);

  // imperative draw loop — reads t/hover from the store without re-rendering React per frame
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !graph || !packIndex) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let lastKey = "";

    function draw() {
      raf = requestAnimationFrame(draw);
      const st = useWorkbench.getState();
      const layer = layerForT(st.t, graph!.num_layers);
      const active = st.hover ?? st.pinned[st.pinned.length - 1] ?? null;
      const res = active ? resolve(active, packIndex!, layer) : null;
      const key = `${mode}|${layer}|${res ? res.node : ""}|${canvas!.clientWidth}x${canvas!.clientHeight}`;
      if (key === lastKey) return; // redraw only on change
      lastKey = key;

      const dpr = window.devicePixelRatio || 1;
      const w = canvas!.clientWidth, h = canvas!.clientHeight;
      canvas!.width = w * dpr;
      canvas!.height = h * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx!.clearRect(0, 0, w, h);
      const L = graph!.layers[layer];
      const grid = graph!.grid;

      if (mode === "layer") {
        for (const [s, d, wt] of L.edges) {
          const [x1, y1] = nodeXY(s, grid, w, h);
          const [x2, y2] = nodeXY(d, grid, w, h);
          ctx!.strokeStyle = `rgba(90,102,122,${Math.min(0.32, wt * 1.4)})`;
          ctx!.lineWidth = 0.4 + wt * 1.6;
          ctx!.beginPath();
          ctx!.moveTo(x1, y1);
          ctx!.lineTo(x2, y2);
          ctx!.stroke();
        }
        for (const n of L.nodes) {
          const [x, y] = nodeXY(n.idx, grid, w, h);
          const hue = HUES[n.community % HUES.length];
          const hit = res?.idx === n.idx && res.layer === layer;
          ctx!.fillStyle = hit ? "#0f1723" : `hsl(${hue} 62% ${n.kind === "cls_token" ? 46 : 58}% / 0.92)`;
          ctx!.beginPath();
          ctx!.arc(x, y, hit ? 5 : n.kind === "cls_token" ? 4 : 2.4, 0, Math.PI * 2);
          ctx!.fill();
          if (hit) {
            ctx!.strokeStyle = "#3b82f6";
            ctx!.lineWidth = 1.4;
            ctx!.beginPath();
            ctx!.arc(x, y, 8, 0, Math.PI * 2);
            ctx!.stroke();
          }
        }
      } else {
        // unrolled: x = layer stratum, y = token idx
        const padX = 24, padY = 8;
        const colW = (w - 2 * padX) / (graph!.num_layers - 1 || 1);
        const yOf = (idx: number) => padY + (idx / (graph!.num_tokens - 1)) * (h - 2 * padY);
        // residual path for the selected token only (per residual.materialized=false convention)
        if (res && res.idx >= 0) {
          ctx!.strokeStyle = "rgba(59,130,246,0.55)";
          ctx!.lineWidth = 1;
          ctx!.beginPath();
          ctx!.moveTo(padX, yOf(res.idx));
          ctx!.lineTo(padX + (graph!.num_layers - 1) * colW, yOf(res.idx));
          ctx!.stroke();
        }
        for (let l = 0; l < graph!.num_layers; l++) {
          const x = padX + l * colW;
          const cur = l === layer;
          for (const n of graph!.layers[l].nodes) {
            const hue = HUES[n.community % HUES.length];
            const hit = res?.idx === n.idx;
            ctx!.fillStyle = hit
              ? "#0f1723"
              : `hsl(${hue} 60% 56% / ${cur ? 0.95 : 0.28})`;
            const r = hit ? 2.5 : cur ? 1.6 : 1;
            ctx!.fillRect(x - r / 2, yOf(n.idx) - r / 2, r, r);
          }
          if (cur) {
            ctx!.strokeStyle = "rgba(59,130,246,0.45)";
            ctx!.strokeRect(x - 4, padY - 2, 8, h - 2 * padY + 4);
          }
        }
      }
    }
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [graph, packIndex, mode]);

  function pick(e: React.MouseEvent): { layer: number; idx: number } | null {
    const canvas = canvasRef.current;
    if (!canvas || !graph) return null;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const st = useWorkbench.getState();
    const layer = layerForT(st.t, graph.num_layers);
    if (mode === "layer") {
      let best: number | null = null, bd = 144;
      for (const n of graph.layers[layer].nodes) {
        const [x, y] = nodeXY(n.idx, graph.grid, r.width, r.height);
        const d = (x - mx) ** 2 + (y - my) ** 2;
        if (d < bd) { bd = d; best = n.idx; }
      }
      return best == null ? null : { layer, idx: best };
    }
    const padX = 24, padY = 8;
    const colW = (r.width - 2 * padX) / (graph.num_layers - 1 || 1);
    const l = Math.max(0, Math.min(graph.num_layers - 1, Math.round((mx - padX) / colW)));
    const idx = Math.max(0, Math.min(graph.num_tokens - 1,
      Math.round(((my - padY) / (r.height - 2 * padY)) * (graph.num_tokens - 1))));
    return { layer: l, idx };
  }

  if (!graph) {
    return (
      <div className="flex h-full items-center justify-center text-[11px] tracking-widest text-muted">
        {manifest ? "no graph asset" : "awaiting pack"}
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        className="h-full w-full"
        onMouseMove={(e) => {
          const p = pick(e);
          setHover(p ? { kind: "node", id: nodeId(p.layer, p.idx) } : null);
        }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const p = pick(e);
          if (p) togglePin({ kind: "node", id: nodeId(p.layer, p.idx) });
        }}
      />
      <div className="absolute right-2 top-2 flex gap-1">
        {(["layer", "unrolled"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-widest ${
              mode === m ? "border-graph/60 bg-graph/10 text-graph" : "border-edge text-muted hover:text-readout"
            }`}
          >
            {m}
          </button>
        ))}
      </div>
      <div className="pointer-events-none absolute bottom-1 left-2 text-[9px] tracking-widest text-muted/70">
        top-{graph.k} attention edges · hue = community
      </div>
    </div>
  );
}
