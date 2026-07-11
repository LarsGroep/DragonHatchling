"use client";

/**
 * Run header — the dataset/model summary and the frozen-feature metric row
 * (linear probe, k-NN vs chance, SOM QE/TE/dead fraction, trustworthiness).
 * Framing stays honest: probe numbers are labelled as SSL yardsticks, and
 * probe/kNN are shown against chance so a reader never mistakes them for
 * supervised accuracy.
 */
import type { UmtvitBundle } from "@/src/lib/umtvit";
import { Metric, fmtNum } from "./controls";

export function RunHeader({ bundle }: { bundle: UmtvitBundle }) {
  const { dataset, model, metrics } = bundle;
  const chance = metrics.chance;
  const probeTone =
    metrics.linear_probe !== null && chance !== null && metrics.linear_probe > chance
      ? "evidence"
      : "readout";
  const knnTone =
    metrics.knn !== null && chance !== null && metrics.knn > chance ? "evidence" : "readout";

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
          hint={chance !== null ? `chance ${fmtNum(chance)}` : "unlabeled"}
          tone={probeTone}
        />
        <Metric
          label="k-NN (cos)"
          value={fmtNum(metrics.knn)}
          hint={chance !== null ? `chance ${fmtNum(chance)}` : "unlabeled"}
          tone={knnTone}
        />
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
        Frozen-feature SSL yardsticks — probe and k-NN are shown against chance, not as supervised
        accuracy. Representation learning used no labels; labels enter only these read-outs.
      </p>
    </section>
  );
}
