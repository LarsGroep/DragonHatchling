"use client";

/**
 * LensReadout — the three honest readings for one lesion, under the active lens
 * (docs/MALIGNANCY-LENS.md). Every number traces to a measurement; the copy
 * never says "stage" or "diagnosis". When the manifold reading is out of
 * distribution, the panel REFUSES rather than asserting — the honest gate.
 */
import { useMemo } from "react";
import type { LensBundle, LensLesion } from "@/src/lib/lens";
import {
  categoryLevels,
  expectedCategory,
  malignantIndices,
  malignantProbability,
  projectFeature,
} from "@/src/lib/malignancy";

export type LensMode = "malignancy" | "category" | "manifold";

const BENIGN = "#0d9488"; // teal
const MALIGN = "#e11d48"; // rose

function mix(a: string, b: string, t: number): string {
  const pa = [parseInt(a.slice(1, 3), 16), parseInt(a.slice(3, 5), 16), parseInt(a.slice(5, 7), 16)];
  const pb = [parseInt(b.slice(1, 3), 16), parseInt(b.slice(3, 5), 16), parseInt(b.slice(5, 7), 16)];
  const c = pa.map((x, i) => Math.round(x + (pb[i] - x) * Math.max(0, Math.min(1, t))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function LensReadout({
  bundle,
  lesion,
  mode,
  threshold,
}: {
  bundle: LensBundle;
  lesion: LensLesion;
  mode: LensMode;
  /** high-sensitivity malignancy decision threshold (default well below 0.5). */
  threshold: number;
}) {
  const malIdx = useMemo(() => malignantIndices(bundle.class_names, bundle.taxonomy), [bundle]);
  const levels = useMemo(() => categoryLevels(bundle.class_names, bundle.taxonomy), [bundle]);

  const pMal = malignantProbability(lesion.probabilities, malIdx);
  const coord = expectedCategory(lesion.probabilities, levels); // [0, K-1]
  const labels = bundle.taxonomy.category_labels;
  const maxLevel = labels.length - 1;
  const reading = bundle.axis ? projectFeature(lesion.feature, bundle.axis) : null;

  // top diagnosis (context only — the lens is a reading OF this, not a diagnosis).
  const topI = lesion.probabilities.reduce((b, p, i, a) => (p > a[b] ? i : b), 0);
  const topClass = bundle.class_names[topI];

  return (
    <div className="flex flex-col gap-3">
      {mode === "malignancy" ? (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] uppercase tracking-wide text-muted">
              model malignancy probability
            </span>
            <span
              className="font-mono text-[20px] font-semibold tabular-nums"
              style={{ color: mix(BENIGN, MALIGN, pMal) }}
            >
              {(pMal * 100).toFixed(1)}%
            </span>
          </div>
          <div className="relative h-2 overflow-hidden rounded-full bg-panel-hi">
            <div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{ width: `${pMal * 100}%`, backgroundColor: mix(BENIGN, MALIGN, pMal) }}
            />
            {/* the high-sensitivity flag threshold */}
            <div
              className="absolute inset-y-0 w-px bg-signal/70"
              style={{ left: `${threshold * 100}%` }}
              title={`flag threshold ${(threshold * 100).toFixed(0)}%`}
            />
          </div>
          <span className="text-[11px] text-muted">
            {pMal >= threshold ? (
              <span className="font-medium text-[#e11d48]">
                above the high-sensitivity flag threshold ({(threshold * 100).toFixed(0)}%) — would
                prompt review
              </span>
            ) : (
              <>below the flag threshold ({(threshold * 100).toFixed(0)}%)</>
            )}{" "}
            · Σ over {malIdx.length} malignant classes · not a diagnosis
          </span>
        </div>
      ) : null}

      {mode === "category" ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-[11px] uppercase tracking-wide text-muted">
            category axis (benign → in-situ → invasive)
          </span>
          <div className="flex items-center gap-1">
            {labels.map((lab, i) => (
              <div key={lab} className="flex flex-1 flex-col items-center gap-1">
                <div
                  className="h-2 w-full rounded-full"
                  style={{
                    backgroundColor:
                      i <= Math.round(coord) ? mix(BENIGN, MALIGN, i / maxLevel) : "#e3e6ec",
                  }}
                />
                <span
                  className={`text-[10px] ${Math.round(coord) === i ? "font-semibold text-readout" : "text-muted"}`}
                >
                  {lab}
                </span>
              </div>
            ))}
          </div>
          <span className="text-[11px] text-muted">
            softmax-weighted position <span className="font-mono">{coord.toFixed(2)}</span> / {maxLevel}
            {" "}· a coarse category reading, <span className="font-medium">not clinical staging</span>
          </span>
        </div>
      ) : null}

      {mode === "manifold" ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-[11px] uppercase tracking-wide text-muted">
            position on the learned benign↔malignant manifold
          </span>
          {reading === null ? (
            <span className="text-[12px] text-muted">no manifold axis in this bundle</span>
          ) : reading.ood ? (
            <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-[12px] text-amber-800">
              <span className="font-semibold">Outside training distribution.</span> This lesion&rsquo;s
              feature sits far off the learned axis (residual{" "}
              <span className="font-mono">{reading.residual.toFixed(2)}</span> &gt; threshold{" "}
              <span className="font-mono">{bundle.axis!.residual_threshold.toFixed(2)}</span>) — the
              lens declines to place it rather than guess.
            </div>
          ) : (
            <>
              <div className="relative h-2 overflow-hidden rounded-full" style={{ background: `linear-gradient(90deg, ${BENIGN}, ${MALIGN})` }}>
                <div
                  className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow"
                  style={{ left: `${reading.position * 100}%`, backgroundColor: mix(BENIGN, MALIGN, reading.position) }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-muted">
                <span>benign</span>
                <span className="font-mono text-readout">{reading.position.toFixed(2)}</span>
                <span>malignant</span>
              </div>
              <span className="text-[11px] text-muted">
                unsupervised projection of the CLS feature · residual{" "}
                <span className="font-mono">{reading.residual.toFixed(2)}</span> (in distribution)
              </span>
            </>
          )}
        </div>
      ) : null}

      <div className="border-t border-edge pt-2 text-[11px] text-muted">
        top diagnosis (context): <span className="font-medium text-readout">{topClass}</span>
        {lesion.true_label ? <> · labelled <span className="text-readout">{lesion.true_label}</span></> : null}
      </div>
    </div>
  );
}
