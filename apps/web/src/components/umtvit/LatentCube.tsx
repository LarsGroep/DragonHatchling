"use client";

/**
 * Latent Cube Explorer — the notebook's "glide through learned depth"
 * animation, made interactive. Pick a probe image, scrub the Z axis (encoder
 * depth) with linear interpolation between slices, play it, and toggle between
 * the untrained (initial) and trained (final) latent volume. Rendered with a
 * magma colormap on a canvas; no image/plot dependency.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { UmtvitBundle, UmtvitCube } from "@/src/lib/umtvit";
import { magma } from "./colormaps";
import { Panel, PlayScrubber } from "./controls";

const CANVAS_PX = 260;

/** Bilinear-in-Z blend of two [H][W] slices at fraction f, drawn with magma. */
function drawSlice(canvas: HTMLCanvasElement, cube: UmtvitCube, z: number) {
  const L = cube.length;
  const lo = Math.max(0, Math.min(L - 1, Math.floor(z)));
  const hi = Math.min(L - 1, lo + 1);
  const f = z - lo;
  const a = cube[lo];
  const b = cube[hi];
  const H = a.length;
  const W = a[0]?.length ?? 0;
  if (!H || !W) return;

  const off = document.createElement("canvas");
  off.width = W;
  off.height = H;
  const octx = off.getContext("2d");
  if (!octx) return;
  const img = octx.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = (1 - f) * a[y][x] + f * (b[y]?.[x] ?? a[y][x]);
      const [r, g, bl] = magma(v);
      const k = (y * W + x) * 4;
      img.data[k] = r;
      img.data[k + 1] = g;
      img.data[k + 2] = bl;
      img.data[k + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, W, H, 0, 0, canvas.width, canvas.height);
}

export function LatentCube({ bundle }: { bundle: UmtvitBundle }) {
  const [probeIdx, setProbeIdx] = useState(0);
  const [z, setZ] = useState(0);
  const [state, setState] = useState<"final" | "initial">("final");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const probe = bundle.probes[Math.min(probeIdx, bundle.probes.length - 1)];
  const cube = state === "final" ? probe?.cube_final : probe?.cube_initial;
  const L = cube?.length ?? 1;
  const maxZ = Math.max(0, L - 1);

  useEffect(() => {
    if (canvasRef.current && cube) drawSlice(canvasRef.current, cube, z);
  }, [cube, z]);

  // clamp z when switching to a probe/state with fewer slices
  useEffect(() => {
    setZ((prev) => Math.min(prev, maxZ));
  }, [maxZ]);

  const legend = useMemo(() => {
    const stops = [0, 0.25, 0.5, 0.75, 1];
    return `linear-gradient(90deg, ${stops
      .map((s) => {
        const [r, g, b] = magma(s);
        return `rgb(${r},${g},${b}) ${s * 100}%`;
      })
      .join(", ")})`;
  }, []);

  if (!bundle.probes.length) {
    return (
      <Panel title="Latent Cube Explorer" accent="latent">
        <p className="text-[12px] text-muted">No probe images in this run.</p>
      </Panel>
    );
  }

  return (
    <Panel
      title="Latent Cube Explorer"
      accent="latent"
      subtitle="Z = encoder depth (learned hierarchy)"
      right={
        <div className="flex items-center rounded-md border border-edge bg-void p-0.5">
          {(["initial", "final"] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setState(s)}
              aria-pressed={state === s}
              className={`rounded px-2 py-0.5 text-[10px] font-medium capitalize transition-colors ${
                state === s ? "bg-latent/10 text-latent" : "text-muted hover:text-readout"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      }
    >
      <div className="flex flex-col gap-3 sm:flex-row">
        {/* probe selector */}
        <div className="flex shrink-0 gap-1.5 sm:flex-col">
          {bundle.probes.map((p, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setProbeIdx(i)}
              aria-pressed={i === probeIdx}
              title={p.label ?? `image ${i}`}
              className={`overflow-hidden rounded-md border transition-colors ${
                i === probeIdx ? "border-latent" : "border-edge hover:border-muted"
              }`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${p.image_png_b64}`}
                alt={p.label ?? `probe ${i}`}
                width={40}
                height={40}
                className="block h-10 w-10 object-cover"
                style={{ imageRendering: "pixelated" }}
              />
            </button>
          ))}
        </div>

        {/* cube canvas + scrubber */}
        <div className="flex min-w-0 flex-1 flex-col items-center gap-2">
          <canvas
            ref={canvasRef}
            width={CANVAS_PX}
            height={CANVAS_PX}
            className="w-full max-w-[260px] rounded-lg border border-edge bg-void"
            style={{ aspectRatio: "1 / 1", imageRendering: "auto" }}
          />
          <div className="flex w-full items-center gap-2">
            <div
              className="h-2 w-16 shrink-0 rounded-full border border-edge"
              style={{ background: legend }}
              title="magma colormap (low → high slice activation)"
            />
            <div className="min-w-0 flex-1">
              <PlayScrubber
                value={z}
                min={0}
                max={maxZ}
                step={0.02}
                speed={1.2}
                accent="latent"
                onChange={setZ}
                format={(v) => `z ${v.toFixed(1)}/${maxZ}`}
              />
            </div>
          </div>
          <p className="text-[11px] text-muted">
            {probe?.label ? `${probe.label} · ` : ""}
            channel-mean, per-slice normalized — shallow z carries detail, deep z carries coarse
            structure if ordering emerged.
          </p>
        </div>
      </div>
    </Panel>
  );
}
