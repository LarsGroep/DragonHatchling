/**
 * Small perceptual colormaps (magma, viridis, hot) as anchor-stop ramps with
 * linear RGB interpolation — enough to reproduce the notebook's matplotlib
 * renderings on a canvas without pulling in a chart/colormap dependency.
 */

export type RGB = [number, number, number];
type Ramp = RGB[];

// 9-stop approximations of the matplotlib maps (sampled at 0, .125 … 1).
const MAGMA: Ramp = [
  [0, 0, 4],
  [24, 15, 62],
  [59, 15, 112],
  [98, 25, 128],
  [140, 41, 129],
  [183, 55, 121],
  [222, 73, 104],
  [246, 148, 100],
  [252, 253, 191],
];

const VIRIDIS: Ramp = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [110, 206, 88],
  [253, 231, 37],
];

const HOT: Ramp = [
  [10, 0, 0],
  [90, 0, 0],
  [170, 0, 0],
  [240, 20, 0],
  [255, 90, 0],
  [255, 160, 0],
  [255, 215, 40],
  [255, 240, 150],
  [255, 255, 255],
];

function sample(ramp: Ramp, t: number): RGB {
  const x = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
  const i = Math.floor(x);
  const j = Math.min(i + 1, ramp.length - 1);
  const f = x - i;
  const a = ramp[i];
  const b = ramp[j];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export const magma = (t: number): RGB => sample(MAGMA, t);
export const viridis = (t: number): RGB => sample(VIRIDIS, t);
export const hot = (t: number): RGB => sample(HOT, t);

export function rgbCss([r, g, b]: RGB): string {
  return `rgb(${r},${g},${b})`;
}

/**
 * Discrete categorical palette echoing matplotlib's tab10 — used to color the
 * embedding scatter by class. Wraps if there are more classes than entries.
 */
export const TAB10: string[] = [
  "#4c78a8",
  "#f58518",
  "#54a24b",
  "#e45756",
  "#72b7b2",
  "#eeca3b",
  "#b279a2",
  "#ff9da6",
  "#9d755d",
  "#bab0ac",
];

export const classColor = (i: number): string => TAB10[((i % TAB10.length) + TAB10.length) % TAB10.length];
