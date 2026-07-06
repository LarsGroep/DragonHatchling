"use client";

/**
 * Latent Embedding Explorer (§10), per-image scope. A fixed PCA basis is fit
 * once on the pack's pooled token embeddings (all 13 steps), then every step
 * projects into it — the cloud animates coherently as t scrubs and the CLS
 * trajectory (comet trail) lives in the same space. Fractional t LERPs
 * between bracketing steps. Dataset-level UMAP/t-SNE landscapes (precomputed
 * projections table) plug into this same pane when a deployment provides
 * them; in mock mode the per-image PCA is what ships. Hover emits token
 * EntityRefs; selections highlight.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { LoadedTokens } from "@/src/lib/pack/types";
import { useWorkbench, layerForT } from "@/src/lib/state/store";
import { resolve } from "@/src/lib/state/resolver";
import { fitPca2, normalizeCoords, project2 } from "../shared/pca";

export function EmbeddingView() {
  const client = useWorkbench((s) => s.client);
  const manifest = useWorkbench((s) => s.manifest);
  const packIndex = useWorkbench((s) => s.packIndex);
  const setHover = useWorkbench((s) => s.setHover);
  const togglePin = useWorkbench((s) => s.togglePin);
  const [tokens, setTokens] = useState<LoadedTokens | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const coordsRef = useRef<Float32Array | null>(null); // [S][T][2] normalized

  useEffect(() => {
    let alive = true;
    setTokens(null);
    coordsRef.current = null;
    if (!client || !manifest) return;
    client.loadTokens(manifest).then((tk) => alive && setTokens(tk)).catch(() => alive && setTokens(null));
    return () => {
      alive = false;
    };
  }, [client, manifest]);

  // fit basis + project all steps once per pack
  const ready = useMemo(() => {
    if (!tokens) return false;
    const { data, steps, tokens: T, dim } = tokens;
    const basis = fitPca2(data, steps * T, dim);
    const raw = project2(data, steps * T, dim, basis);
    coordsRef.current = normalizeCoords(raw);
    return true;
  }, [tokens]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !ready || !tokens || !packIndex) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let lastKey = "";
    const T = tokens.tokens, S = tokens.steps;

    function xyAt(step: number, idx: number, w: number, h: number): [number, number] {
      const c = coordsRef.current!;
      const o = (step * T + idx) * 2;
      return [c[o] * w, c[o + 1] * h];
    }

    function draw() {
      raf = requestAnimationFrame(draw);
      const st = useWorkbench.getState();
      const t = Math.max(0, Math.min(S - 1, st.t));
      const s0 = Math.floor(t), s1 = Math.min(S - 1, s0 + 1), f = t - s0;
      const active = st.hover ?? st.pinned[st.pinned.length - 1] ?? null;
      const res = active ? resolve(active, packIndex!, layerForT(st.t, S - 1)) : null;
      const key = `${t.toFixed(3)}|${res ? res.idx : ""}|${canvas!.clientWidth}`;
      if (key === lastKey) return;
      lastKey = key;

      const dpr = window.devicePixelRatio || 1;
      const w = canvas!.clientWidth, h = canvas!.clientHeight;
      canvas!.width = w * dpr;
      canvas!.height = h * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx!.clearRect(0, 0, w, h);

      // CLS trajectory comet trail
      ctx!.strokeStyle = "rgba(87,227,137,0.35)";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      for (let s = 0; s < S; s++) {
        const [x, y] = xyAt(s, 0, w, h);
        s ? ctx!.lineTo(x, y) : ctx!.moveTo(x, y);
      }
      ctx!.stroke();

      for (let i = 0; i < T; i++) {
        const [x0, y0] = xyAt(s0, i, w, h);
        const [x1, y1] = xyAt(s1, i, w, h);
        const x = x0 + (x1 - x0) * f, y = y0 + (y1 - y0) * f;
        const isCls = i === 0;
        const hit = res != null && (res.idx === i || res.refs.some((r) => r.kind === "token" && r.idx === i));
        ctx!.fillStyle = hit ? "#e8eefc" : isCls ? "#57e389" : "rgba(87,227,137,0.45)";
        ctx!.beginPath();
        ctx!.arc(x, y, hit ? 4 : isCls ? 3.5 : 1.8, 0, Math.PI * 2);
        ctx!.fill();
        if (hit || isCls) {
          ctx!.strokeStyle = hit ? "#e8eefc" : "rgba(87,227,137,0.8)";
          ctx!.beginPath();
          ctx!.arc(x, y, hit ? 7 : 5.5, 0, Math.PI * 2);
          ctx!.stroke();
        }
      }
    }
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [ready, tokens, packIndex]);

  function pick(e: React.MouseEvent): number | null {
    const canvas = canvasRef.current;
    const c = coordsRef.current;
    if (!canvas || !c || !tokens) return null;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const st = useWorkbench.getState();
    const s = Math.round(Math.max(0, Math.min(tokens.steps - 1, st.t)));
    let best: number | null = null, bd = 100;
    for (let i = 0; i < tokens.tokens; i++) {
      const o = (s * tokens.tokens + i) * 2;
      const d = (c[o] * r.width - mx) ** 2 + (c[o + 1] * r.height - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  if (!ready) {
    return (
      <div className="flex h-full items-center justify-center text-[11px] tracking-widest text-muted">
        {manifest ? "projecting…" : "awaiting pack"}
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        className="h-full w-full"
        onMouseMove={(e) => {
          const i = pick(e);
          const st = useWorkbench.getState();
          setHover(i == null ? null : { kind: "token", layer: layerForT(st.t, (tokens?.steps ?? 13) - 1), idx: i });
        }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const i = pick(e);
          const st = useWorkbench.getState();
          if (i != null) togglePin({ kind: "token", layer: layerForT(st.t, (tokens?.steps ?? 13) - 1), idx: i });
        }}
      />
      <div className="pointer-events-none absolute bottom-1 left-2 text-[9px] tracking-widest text-muted/70">
        PCA (pooled 13-step basis) · trail = CLS trajectory
      </div>
    </div>
  );
}
