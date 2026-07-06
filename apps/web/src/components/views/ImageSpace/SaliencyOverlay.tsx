"use client";

/**
 * Per-method saliency overlay (§13): draws the 14×14 attribution grid at the
 * current layer as a soft luminous heatmap over the image. The 14×14 canvas is
 * CSS-upscaled (bilinear) so the map reads as a smooth field.
 */
import { useEffect, useRef } from "react";
import type { LoadedAttribution } from "@/src/lib/pack/types";
import { attributionGrid, magma } from "./saliency";

export function SaliencyOverlay({
  attribution,
  grid,
  layer,
  opacity,
}: {
  attribution: LoadedAttribution;
  grid: number;
  layer: number;
  opacity: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const values = attributionGrid(attribution, grid, layer);
    const img = ctx.createImageData(grid, grid);
    for (let i = 0; i < grid * grid; i++) {
      const [r, g, b, a] = magma(values[i]);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = a;
    }
    ctx.putImageData(img, 0, 0);
  }, [attribution, grid, layer]);

  return (
    <canvas
      ref={ref}
      width={grid}
      height={grid}
      aria-hidden
      className="pointer-events-none absolute inset-0 h-full w-full mix-blend-screen"
      style={{ imageRendering: "auto", opacity }}
    />
  );
}
