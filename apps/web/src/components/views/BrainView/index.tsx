"use client";

/**
 * BrainView (UX-VISION-2) — the living Hebbian brain, the identity of ViTreous.
 *
 * A force-directed, Obsidian-style graph (canvas-2D) over the pack's graph.json:
 * the LAST layer's attention edges are the resting "memory" topology, gravity
 * groups pull real communities into clusters, and the layout is precomputed on
 * pack load then kept alive by a gentle low-alpha drift so the brain breathes
 * even before inference.
 *
 * ACTIVATION (driven by the loop clock `t`, read imperatively — zero React
 * re-renders per frame): at each t, a node's target activation is its token's
 * L2 activation magnitude at layer t (tokens.bin norms, normalized per layer);
 * an EMA blends it across frames so nodes brighten/fade gradually. Only the top
 * ~15% stay "hot". Edges whose endpoints are both active illuminate softly with
 * small traveling pulses (strongest ~40 only). As t→12 nodes with high final
 * Chefer attribution blend toward green — evidence confirmed, one region wins.
 *
 * Community labels fade in near a cluster's centroid only while its mean
 * activation crosses a threshold, and are drawn from REAL data (honesty rule):
 * dominant firing SAE feature id if a plurality exists, else the geometric
 * region descriptor. Hover/click emit {kind:"node"} EntityRefs exactly like the
 * old graph; store selections light the matching node. An Expert toggle reveals
 * the old per-layer grid / unrolled renderer + graph stats (default = brain).
 */
import { useEffect, useRef, useState } from "react";
import type { GraphJson, LoadedAttribution, LoadedTokens } from "@/src/lib/pack/types";
import { useWorkbench, layerForT } from "@/src/lib/state/store";
import { resolve } from "@/src/lib/state/resolver";
import { nodeId } from "@/src/lib/state/packIndex";
import { verdictProgress } from "@/src/lib/loop/schedule";
import { GraphView } from "../GraphView";
import {
  buildBrainGraph,
  centroid,
  computeLayout,
  forceStep,
  layoutBounds,
  type BrainGraph,
} from "./force";
import {
  communityLabel,
  controlPoint,
  ema,
  emaInto,
  hotThreshold,
  meanOver,
  normalizeUnit,
  quadPoint,
  shouldShowLabel,
  tokenNorms,
} from "./activation";

// -- palette (mirrors tailwind light theme) --------------------------------- //
const QUIET: RGB = [138, 148, 164]; // resting node gray (visible on white)
const ACTIVE: RGB = [59, 130, 246]; // soft blue — activity
const EVIDENCE: RGB = [34, 197, 94]; // green — confirmed evidence
const EDGE_BASE = "rgba(88,99,120,0.16)";
const LABEL_TEXT: RGB = [40, 49, 63];

const LABEL_THRESHOLD = 0.42;
const HOT_FRACTION = 0.15;
const PULSE_EDGES = 40;
const ACT_EMA = 0.09; // per-frame activation blend
const DRIFT_ALPHA = 0.012; // continuous "alive" sim rate

type RGB = [number, number, number];

function mix(a: RGB, b: RGB, u: number): RGB {
  const c = Math.max(0, Math.min(1, u));
  return [a[0] + (b[0] - a[0]) * c, a[1] + (b[1] - a[1]) * c, a[2] + (b[2] - a[2]) * c];
}
function rgba(c: RGB, alpha: number): string {
  return `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${alpha})`;
}

interface View {
  k: number;
  panx: number;
  pany: number;
}

interface LabelState {
  vis: boolean;
  alpha: number;
  text: string;
}

export function BrainView() {
  const client = useWorkbench((s) => s.client);
  const manifest = useWorkbench((s) => s.manifest);
  const packIndex = useWorkbench((s) => s.packIndex);
  const setHover = useWorkbench((s) => s.setHover);
  const togglePin = useWorkbench((s) => s.togglePin);

  const [graph, setGraph] = useState<GraphJson | null>(null);
  const [tokens, setTokens] = useState<LoadedTokens | null>(null);
  const [chefer, setChefer] = useState<LoadedAttribution | null>(null);
  const [expert, setExpert] = useState(false);
  const [ready, setReady] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Imperative render state (never triggers React re-renders).
  const brainRef = useRef<BrainGraph | null>(null);
  const actRef = useRef<Float32Array | null>(null); // per-node EMA activation
  const targetRef = useRef<Float32Array | null>(null); // per-node target scratch
  const chefRef = useRef<Float32Array | null>(null); // per-node final attribution (0..1)
  const top1Ref = useRef<Record<number, number> | null>(null);
  const labelsRef = useRef<Map<number, LabelState>>(new Map());
  const fitRef = useRef(1);
  const viewRef = useRef<View>({ k: 1, panx: 0, pany: 0 });
  const dragRef = useRef({ down: false, x: 0, y: 0, moved: 0 });

  // -- data loads ----------------------------------------------------------- //
  useEffect(() => {
    let alive = true;
    setGraph(null);
    setTokens(null);
    setChefer(null);
    setReady(false);
    if (!client || !manifest) return;
    if (!("graph.json" in manifest.assets)) return;
    client.loadGraph().then((g) => alive && setGraph(g)).catch(() => {});
    if ("tokens.bin" in manifest.assets) {
      client.loadTokens(manifest).then((t) => alive && setTokens(t)).catch(() => {});
    }
    client
      .loadAttribution("chefer", manifest)
      .then((a) => alive && setChefer(a))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [client, manifest]);

  // Top-1 firing feature per token (for honest community labels).
  useEffect(() => {
    if (!packIndex) {
      top1Ref.current = null;
      return;
    }
    const map: Record<number, number> = {};
    packIndex.tokenConcepts.forEach((ids, idx) => {
      if (ids.length) {
        const f = Number(ids[0]);
        if (Number.isFinite(f)) map[idx] = f;
      }
    });
    top1Ref.current = Object.keys(map).length ? map : null;
  }, [packIndex]);

  // -- precompute the layout on graph load (async, ~1-2s budget) ------------ //
  useEffect(() => {
    setReady(false);
    brainRef.current = null;
    if (!graph) return;
    let alive = true;
    // Yield a frame so the "warming up" state paints, then relax the layout.
    const id = setTimeout(() => {
      if (!alive) return;
      const brain = buildBrainGraph(graph);
      computeLayout(brain, 320);
      brainRef.current = brain;
      actRef.current = new Float32Array(brain.nodes.length);
      targetRef.current = new Float32Array(brain.nodes.length);
      labelsRef.current = new Map();
      // Fit transform from the relaxed bounds.
      const b = layoutBounds(brain.nodes);
      const halfW = Math.max(1e-3, (b.maxX - b.minX) / 2);
      const halfH = Math.max(1e-3, (b.maxY - b.minY) / 2);
      fitRef.current = 0.92 / Math.max(halfW, halfH);
      viewRef.current = { k: 1, panx: 0, pany: 0 };
      setReady(true);
    }, 16);
    return () => {
      alive = false;
      clearTimeout(id);
    };
  }, [graph]);

  // Final-layer Chefer attribution per node (normalized 0..1) for the green blend.
  useEffect(() => {
    const brain = brainRef.current;
    if (!chefer || !brain || !ready) {
      chefRef.current = null;
      return;
    }
    // chefer shape [L, T]: take the last layer row.
    const [L, T] = chefer.shape;
    const row = Math.max(0, L - 1);
    const perToken = new Float32Array(T);
    for (let i = 0; i < T; i++) perToken[i] = chefer.data[row * T + i];
    const norm = normalizeUnit(perToken, 0);
    const perNode = new Float32Array(brain.nodes.length);
    brain.nodes.forEach((n, i) => (perNode[i] = norm[n.idx] ?? 0));
    chefRef.current = perNode;
  }, [chefer, ready]);

  // -- world/screen transform helpers --------------------------------------- //
  const dims = () => {
    const c = canvasRef.current!;
    return { w: c.clientWidth, h: c.clientHeight };
  };
  const worldToScreen = (wx: number, wy: number, w: number, h: number): [number, number] => {
    const v = viewRef.current;
    const s = fitRef.current * v.k * Math.min(w, h) * 0.5;
    return [w / 2 + v.panx + wx * s, h / 2 + v.pany + wy * s];
  };
  const screenToWorld = (sx: number, sy: number, w: number, h: number): [number, number] => {
    const v = viewRef.current;
    const s = fitRef.current * v.k * Math.min(w, h) * 0.5;
    return [(sx - w / 2 - v.panx) / s, (sy - h / 2 - v.pany) / s];
  };

  function nearestNode(sx: number, sy: number): number {
    const brain = brainRef.current;
    if (!brain) return -1;
    const { w, h } = dims();
    let best = -1;
    let bd = 18 * 18;
    brain.nodes.forEach((n, i) => {
      const [x, y] = worldToScreen(n.x, n.y, w, h);
      const d = (x - sx) ** 2 + (y - sy) ** 2;
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  }

  // -- the render loop (imperative; brain mode only) ------------------------- //
  useEffect(() => {
    if (expert) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const resize = () => {
      const el = wrapRef.current;
      if (!el) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = el.clientWidth * dpr;
      canvas.height = el.clientHeight * dpr;
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    if (wrapRef.current) ro.observe(wrapRef.current);

    function frame() {
      raf = requestAnimationFrame(frame);
      const brain = brainRef.current;
      const act = actRef.current;
      const target = targetRef.current;
      if (!brain || !act || !target) return;

      const { w, h } = dims();
      ctx!.clearRect(0, 0, w, h);

      const st = useWorkbench.getState();
      const nLayers = graph?.num_layers ?? 12;
      const t = st.t;

      // gentle continuous drift keeps the brain "alive".
      forceStep(brain.nodes, brain.links, brain.communities, DRIFT_ALPHA);

      // activation target from token norms at the current layer.
      if (tokens) {
        const step = Math.max(0, Math.min(tokens.steps - 1, Math.round(t)));
        const norms = tokenNorms(tokens.data, step, tokens.tokens, tokens.dim);
        const norm01 = normalizeUnit(norms, 0);
        brain.nodes.forEach((n, i) => (target[i] = norm01[n.idx] ?? 0));
      }
      emaInto(act, target, ACT_EMA);

      const gp = verdictProgress(t); // 0 → 1 as t crosses the final stage
      const chef = chefRef.current;
      const thr = hotThreshold(act, HOT_FRACTION);

      // selection → highlighted node idxs.
      const active = st.hover ?? st.pinned[st.pinned.length - 1] ?? null;
      const litIdx = new Set<number>();
      if (active && packIndex) {
        for (const e of resolve(active, packIndex, layerForT(t, nLayers)).refs) {
          if (e.kind === "token" || e.kind === "gaussian") litIdx.add(e.idx);
          if (e.kind === "node") {
            const m = /^L\d+_T(\d+)$/.exec(e.id);
            if (m) litIdx.add(Number(m[1]));
          }
        }
      }

      const now = performance.now() / 1000;

      // -- edges: subtle base + soft illumination for co-active pairs -------- //
      const pulseCandidates: { link: number; activity: number }[] = [];
      brain.links.forEach((lk, i) => {
        const na = brain.nodes[lk.a];
        const nb = brain.nodes[lk.b];
        const [x1, y1] = worldToScreen(na.x, na.y, w, h);
        const [x2, y2] = worldToScreen(nb.x, nb.y, w, h);
        const [cx, cy] = controlPoint(x1, y1, x2, y2, 0.14);
        const activity = Math.min(act[lk.a], act[lk.b]);
        ctx!.beginPath();
        ctx!.moveTo(x1, y1);
        ctx!.quadraticCurveTo(cx, cy, x2, y2);
        if (activity > thr * 0.7) {
          ctx!.strokeStyle = rgba(ACTIVE, Math.min(0.5, activity * 0.7));
          ctx!.lineWidth = 0.8 + activity * 1.6;
          pulseCandidates.push({ link: i, activity });
        } else {
          ctx!.strokeStyle = EDGE_BASE;
          ctx!.lineWidth = 0.8;
        }
        ctx!.stroke();
      });

      // -- traveling pulses on the strongest ~40 active edges ---------------- //
      pulseCandidates.sort((a, b) => b.activity - a.activity);
      const top = pulseCandidates.slice(0, PULSE_EDGES);
      for (let ti = 0; ti < top.length; ti++) {
        const lk = brain.links[top[ti].link];
        const na = brain.nodes[lk.a];
        const nb = brain.nodes[lk.b];
        const [x1, y1] = worldToScreen(na.x, na.y, w, h);
        const [x2, y2] = worldToScreen(nb.x, nb.y, w, h);
        const [cx, cy] = controlPoint(x1, y1, x2, y2, 0.14);
        const phase = (ti * 0.137) % 1;
        for (let d = 0; d < 2; d++) {
          const u = (now * 0.8 + phase + d * 0.5) % 1;
          const [px, py] = quadPoint(x1, y1, cx, cy, x2, y2, u);
          ctx!.beginPath();
          ctx!.arc(px, py, 1.9, 0, Math.PI * 2);
          ctx!.fillStyle = rgba(ACTIVE, 0.85 * (1 - Math.abs(u - 0.5) * 0.6));
          ctx!.fill();
        }
      }

      // -- nodes ------------------------------------------------------------- //
      brain.nodes.forEach((n, i) => {
        const a = act[i];
        const hot = a >= thr;
        const g = chef ? chef[i] * gp : 0;
        // resting gray → soft blue by activation → green by confirmed evidence.
        const base = mix(QUIET, ACTIVE, hot ? 0.35 + a * 0.65 : a * 0.5);
        const col = mix(base, EVIDENCE, g);
        const [x, y] = worldToScreen(n.x, n.y, w, h);
        const r = (n.kind === "cls_token" ? 3.4 : 2.2) + a * 3.2 + g * 1.2;
        const lit = litIdx.has(n.idx);
        // soft activity halo for hot nodes.
        if (hot || g > 0.2) {
          ctx!.beginPath();
          ctx!.arc(x, y, r + 4 + a * 4, 0, Math.PI * 2);
          ctx!.fillStyle = rgba(g > 0.2 ? EVIDENCE : ACTIVE, 0.06 + a * 0.10);
          ctx!.fill();
        }
        ctx!.beginPath();
        ctx!.arc(x, y, lit ? r + 1.5 : r, 0, Math.PI * 2);
        ctx!.fillStyle = lit ? "#0f1723" : rgba(col, 0.72 + a * 0.28);
        ctx!.fill();
        if (lit) {
          ctx!.strokeStyle = rgba(ACTIVE, 0.9);
          ctx!.lineWidth = 1.6;
          ctx!.beginPath();
          ctx!.arc(x, y, r + 5, 0, Math.PI * 2);
          ctx!.stroke();
        }
      });

      // -- community labels (fade in while active; honest names) ------------- //
      const grid = packIndex?.grid ?? graph?.grid ?? 14;
      ctx!.textAlign = "center";
      ctx!.textBaseline = "middle";
      ctx!.font = "600 12px var(--font-sans), system-ui, sans-serif";
      for (const [id, members] of brain.communities) {
        if (members.length < 3) continue;
        const meanA = meanOver(act, members);
        let ls = labelsRef.current.get(id);
        if (!ls) {
          const tokenIdxs = members.map((mi) => brain.nodes[mi].idx);
          ls = { vis: false, alpha: 0, text: communityLabel(tokenIdxs, grid, top1Ref.current) };
          labelsRef.current.set(id, ls);
        }
        ls.vis = shouldShowLabel(meanA, ls.vis, LABEL_THRESHOLD);
        ls.alpha = ema(ls.alpha, ls.vis ? 1 : 0, 0.08);
        if (ls.alpha < 0.02) continue;
        const c = centroid(brain.nodes, members);
        const [lx, ly] = worldToScreen(c.x, c.y, w, h);
        const tw = ctx!.measureText(ls.text).width;
        ctx!.fillStyle = rgba([255, 255, 255], ls.alpha * 0.74);
        ctx!.fillRect(lx - tw / 2 - 6, ly - 22, tw + 12, 16);
        ctx!.fillStyle = rgba(LABEL_TEXT, ls.alpha);
        ctx!.fillText(ls.text, lx, ly - 14);
      }
    }
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [expert, graph, tokens, packIndex]);

  // -- interaction: hover / click / pan / zoom ------------------------------- //
  function onMove(e: React.PointerEvent) {
    const canvas = canvasRef.current;
    if (!canvas || !ready) return;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    const d = dragRef.current;
    if (d.down) {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      d.moved += Math.abs(dx) + Math.abs(dy);
      d.x = e.clientX;
      d.y = e.clientY;
      viewRef.current.panx += dx;
      viewRef.current.pany += dy;
      return;
    }
    const idx = nearestNode(mx, my);
    const layer = layerForT(useWorkbench.getState().t, graph?.num_layers ?? 12);
    if (idx < 0) {
      if (useWorkbench.getState().hover?.kind === "node") setHover(null);
      return;
    }
    setHover({ kind: "node", id: nodeId(layer, brainRef.current!.nodes[idx].idx) });
  }

  function onDown(e: React.PointerEvent) {
    dragRef.current = { down: true, x: e.clientX, y: e.clientY, moved: 0 };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  }

  function onUp(e: React.PointerEvent) {
    const d = dragRef.current;
    d.down = false;
    if (d.moved < 5 && ready) {
      const canvas = canvasRef.current!;
      const r = canvas.getBoundingClientRect();
      const idx = nearestNode(e.clientX - r.left, e.clientY - r.top);
      if (idx >= 0) {
        const layer = layerForT(useWorkbench.getState().t, graph?.num_layers ?? 12);
        togglePin({ kind: "node", id: nodeId(layer, brainRef.current!.nodes[idx].idx) });
      }
    }
  }

  function onWheel(e: React.WheelEvent) {
    if (!ready) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const { w, h } = dims();
    const before = screenToWorld(mx, my, w, h);
    const factor = Math.exp(-e.deltaY * 0.0012);
    viewRef.current.k = Math.max(0.4, Math.min(6, viewRef.current.k * factor));
    const after = screenToWorld(mx, my, w, h);
    const s = fitRef.current * viewRef.current.k * Math.min(w, h) * 0.5;
    viewRef.current.panx += (after[0] - before[0]) * s;
    viewRef.current.pany += (after[1] - before[1]) * s;
  }

  function resetView() {
    viewRef.current = { k: 1, panx: 0, pany: 0 };
  }

  const hasGraphAsset = !!manifest && "graph.json" in manifest.assets;

  return (
    <div ref={wrapRef} data-testid="brain-view" className="relative h-full w-full overflow-hidden">
      {!expert ? (
        <canvas
          ref={canvasRef}
          className="h-full w-full touch-none cursor-grab active:cursor-grabbing"
          onPointerMove={onMove}
          onPointerDown={onDown}
          onPointerUp={onUp}
          onPointerLeave={() => setHover(null)}
          onWheel={onWheel}
        />
      ) : (
        <div className="h-full w-full">
          <GraphView />
          {graph ? (
            <div className="pointer-events-none absolute bottom-2 left-3 flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-[10px] text-muted">
              <span>{graph.num_tokens} nodes</span>
              <span>{graph.layers[graph.num_layers - 1]?.edges.length ?? 0} edges (L{graph.num_layers - 1})</span>
              <span>{graph.num_layers} layers</span>
              <span>top-{graph.k} attention</span>
              {brainRef.current ? <span>{brainRef.current.communities.size} communities</span> : null}
            </div>
          ) : null}
        </div>
      )}

      {/* controls */}
      <div className="absolute right-3 top-3 flex items-center gap-1.5">
        {!expert && ready ? (
          <button
            type="button"
            onClick={resetView}
            title="Reset view"
            className="rounded-md border border-edge bg-void/80 px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-muted shadow-soft backdrop-blur transition-colors hover:text-readout"
          >
            reset
          </button>
        ) : null}
        <div className="flex items-center rounded-md border border-edge bg-void/80 p-0.5 shadow-soft backdrop-blur">
          {(["brain", "expert"] as const).map((m) => {
            const on = (m === "expert") === expert;
            return (
              <button
                key={m}
                type="button"
                onClick={() => setExpert(m === "expert")}
                title={m === "expert" ? "Expert overlay — layer grid, unrolled, graph stats" : "The brain"}
                className={`rounded px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide transition-colors ${
                  on ? "bg-graph/10 text-graph" : "text-muted hover:text-readout"
                }`}
              >
                {m}
              </button>
            );
          })}
        </div>
      </div>

      {/* status / warming */}
      {!hasGraphAsset ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-[11px] tracking-wide text-muted">
          {manifest ? "this pack carries no brain graph" : "awaiting pack"}
        </div>
      ) : !expert && !ready ? (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-graph/70" />
          <span className="text-[11px] tracking-wide">warming up the brain…</span>
        </div>
      ) : null}
    </div>
  );
}
