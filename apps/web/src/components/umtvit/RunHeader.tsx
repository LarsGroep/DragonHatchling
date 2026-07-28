"use client";

/**
 * Run header — the dataset/model summary and the frozen-feature metric row
 * (linear probe, k-NN, SOM QE/TE/dead fraction, trustworthiness).
 *
 * BASELINE HONESTY: this row used to colour a metric as "evidence" whenever it
 * beat UNIFORM chance (1/K). On an imbalanced dataset that is not a bar at all
 * — HAM10000 is ~67% `nv`, so a model that has learned nothing but the class
 * prior scores 0.669 and would still have rendered green against chance 0.143.
 * A metric now reads as evidence only when it beats the MAJORITY-CLASS
 * baseline, and when the run did not record that baseline the metric stays
 * neutral rather than being flattered by the wrong comparison.
 */
import type { UmtvitBundle } from "@/src/lib/umtvit";
import { Metric, fmtNum } from "./controls";

/** Baseline hint text: the majority-class floor is the one that matters. */
function baselineHint(chance: number | null, majority: number | null): string {
  if (majority !== null) {
    return `vs majority ${fmtNum(majority)}${chance !== null ? ` · unif ${fmtNum(chance)}` : ""}`;
  }
  if (chance !== null) return `unif ${fmtNum(chance)} · majority not recorded`;
  return "unlabeled";
}

export function RunHeader({ bundle }: { bundle: UmtvitBundle }) {
  const { dataset, model, metrics } = bundle;
  const chance = metrics.chance;
  const majority = metrics.majority_baseline;
  // Evidence requires clearing the majority-class floor, never uniform chance.
  const beatsBaseline = (m: number | null) =>
    m !== null && majority !== null && m > majority ? "evidence" : "readout";
  const probeTone = beatsBaseline(metrics.linear_probe);
  const knnTone = beatsBaseline(metrics.knn);
  const hint = baselineHint(chance, majority);

  const summary = [
    `dim ${model.dim}`,
    `depth/Z ${model.depth}`,
    `volume ${model.volume_grid}²×${model.depth}×${model.volume_channels}`,
    `SOM ${model.som_grid.join("×")}`,
    `x-attn ${model.cross_attention}`,
    `${model.params_millions.toFixed(2)} M params`,
  ];

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-edge bg-panel p-4 shadow-soft">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-[15px] font-semibold tracking-tight text-signal">{dataset.name}</span>
        <span className="text-[12px] text-muted">
          {dataset.image_size}px · {dataset.augmentation} ·{" "}
          {dataset.num_classes > 0 ? `${dataset.num_classes} classes` : "unlabeled"}
        </span>
        <span className="ml-auto font-mono text-[11px] text-muted">
          {summary.join("  ·  ")}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
        <Metric
          label="Linear probe"
          value={fmtNum(metrics.linear_probe)}
          hint={hint}
          tone={probeTone}
        />
        <Metric label="k-NN (cos)" value={fmtNum(metrics.knn)} hint={hint} tone={knnTone} />
        <Metric label="SOM QE" value={fmtNum(metrics.som_quantization_error)} hint="quantization err" />
        <Metric
          label="SOM TE"
          value={fmtNum(metrics.som_topographic_error)}
          hint="lower = better"
        />
        <Metric
          label="SOM dead"
          value={fmtNum(metrics.som_dead_fraction)}
          hint="dead-neuron frac"
          tone={
            metrics.som_dead_fraction !== null && metrics.som_dead_fraction > 0.5 ? "warm" : "readout"
          }
        />
        <Metric
          label="Trustworth."
          value={fmtNum(metrics.trustworthiness)}
          hint="1.0 = perfect"
        />
        <Metric label="Schema" value={`v${bundle.version}`} hint="umtvit_web.json" tone="muted" />
      </div>

      <p className="text-[11px] leading-relaxed text-muted">
        Frozen-feature SSL yardsticks, not supervised accuracy. Representation learning used no
        labels; labels enter only these read-outs. Read probe and k-NN against the{" "}
        <span className="text-readout">majority-class baseline</span> — on an imbalanced dataset
        (HAM10000 is ~67% nevi) always predicting the commonest class already scores that, so
        uniform chance is not the bar. Accuracy alone is the wrong summary here: per-class recall,
        melanoma sensitivity above all, is the number that matters.
      </p>
    </section>
  );
}
