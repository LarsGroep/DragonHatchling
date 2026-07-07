"use client";

/**
 * Workbench — the client orchestrator (§11–§13). Owns:
 *   • initial data load (datasets → first gallery → first pack),
 *   • the requestAnimationFrame replay clock that advances the timeline `t`
 *     while `playing` (sweeps 0..L then loops),
 *   • global keyboard transport (space / arrows / home / end),
 * and composes the header, gallery strip, the four synchronized panes, and the
 * transport bar. Image Space is live at M5; the other three panes are
 * store-driven SyncedPlaceholders proving the four-way link until M6/M7.
 */
import { useCallback, useEffect, useState } from "react";
import type { DatasetRow, GalleryImageRow } from "@/src/lib/db/types";
import { getDb } from "@/src/lib/db/client";
import { useWorkbench } from "@/src/lib/state/store";
import { LOOP_STAGES, type FeaturePane } from "@/src/lib/loop/schedule";
import { WorkbenchPanel, type PanelAccent } from "@/components/WorkbenchPanel";
import { WorkbenchHeader } from "./WorkbenchHeader";
import { GalleryStrip } from "./GalleryStrip";
import { Transport } from "./Transport";
import { LoopController } from "./LoopController";
import { LoopCaption } from "./LoopCaption";
import { ImageSpaceView } from "./views/ImageSpace";
import { GaussianFieldView } from "./views/GaussianField";
import { GraphView } from "./views/GraphView";
import { EmbeddingView } from "./views/EmbeddingView";

/**
 * The four panes, with Expert (instrument) titles and Plain (lay) titles. The
 * ambient loop spotlights `feature` panes stage-by-stage (S1); `accent` picks
 * the wayfinding hue and the vignette color.
 */
const PANES: {
  accent: PanelAccent;
  feature: FeaturePane;
  milestone: string;
  expertTitle: string;
  plainTitle: string;
}[] = [
  { accent: "image", feature: "image", milestone: "M5", expertTitle: "IMAGE SPACE", plainTitle: "The photo & the evidence" },
  { accent: "gauss", feature: "gauss", milestone: "M6", expertTitle: "GAUSSIAN FEATURE FIELD", plainTitle: "What the model senses" },
  { accent: "graph", feature: "graph", milestone: "M7", expertTitle: "INTERACTION GRAPH", plainTitle: "The model's neurons" },
  { accent: "latent", feature: "latent", milestone: "M7", expertTitle: "LATENT EMBEDDINGS", plainTitle: "The model's map of ideas" },
];

const PANE_CONTENT = (accent: PanelAccent, classNames?: string[]) => {
  switch (accent) {
    case "image":
      return <ImageSpaceView classNames={classNames} />;
    case "gauss":
      return <GaussianFieldView />;
    case "graph":
      return <GraphView />;
    case "latent":
      return <EmbeddingView />;
  }
};

export function Workbench() {
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [activeDataset, setActiveDataset] = useState<DatasetRow | null>(null);
  const [images, setImages] = useState<GalleryImageRow[]>([]);
  const [bootError, setBootError] = useState<string | null>(null);

  const selectImage = useWorkbench((s) => s.selectImage);
  const mode = useWorkbench((s) => s.mode);
  const loopStage = useWorkbench((s) => s.loopStage);
  const featurePane = LOOP_STAGES[Math.min(loopStage, LOOP_STAGES.length - 1)].feature;

  // -- initial load: datasets → first dataset's gallery → first image ------ //
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const db = getDb();
        const ds = await db.listDatasets();
        if (!alive) return;
        setDatasets(ds);
        if (ds.length) {
          const first = ds[0];
          setActiveDataset(first);
          const imgs = await db.listGalleryImages(first.id);
          if (!alive) return;
          setImages(imgs);
          if (imgs.length) await selectImage(first.id, imgs[0]);
        }
      } catch (e) {
        if (alive) setBootError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [selectImage]);

  const onDataset = useCallback(
    async (d: DatasetRow) => {
      setActiveDataset(d);
      try {
        const imgs = await getDb().listGalleryImages(d.id);
        setImages(imgs);
        if (imgs.length) await selectImage(d.id, imgs[0]);
      } catch (e) {
        setBootError(e instanceof Error ? e.message : String(e));
      }
    },
    [selectImage],
  );

  // The ambient replay clock now lives in <LoopController/> (S1).

  // -- global keyboard transport ------------------------------------------ //
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      const st = useWorkbench.getState();
      switch (e.key) {
        case " ":
          e.preventDefault();
          st.togglePlay();
          break;
        case "ArrowLeft":
          st.stepLayer(-1);
          break;
        case "ArrowRight":
          st.stepLayer(1);
          break;
        case "Home":
          st.seekLayer(0);
          break;
        case "End":
          st.seekLayer(st.packIndex?.numLayers ?? 12);
          break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex h-screen flex-col">
      <WorkbenchHeader datasetName={activeDataset?.display_name} />
      <GalleryStrip
        datasets={datasets}
        activeDataset={activeDataset}
        images={images}
        onDataset={onDataset}
      />

      {bootError ? (
        <div className="border-b border-gauss/40 bg-gauss/10 px-4 py-1.5 text-[11px] text-gauss">
          data load failed: {bootError}
        </div>
      ) : null}

      <div className="relative min-h-0 flex-1">
        <main className="grid h-full min-h-0 grid-cols-1 grid-rows-[repeat(4,minmax(180px,1fr))] gap-3 p-3 lg:grid-cols-2 lg:grid-rows-2">
          {PANES.map((p) => (
            <WorkbenchPanel
              key={p.accent}
              title={mode === "plain" ? p.plainTitle : p.expertTitle}
              accent={p.accent}
              milestone={p.milestone}
              hint=""
              featured={featurePane === p.feature}
            >
              <div className="h-full w-full">{PANE_CONTENT(p.accent, activeDataset?.class_names)}</div>
            </WorkbenchPanel>
          ))}
        </main>

        {/* Ambient inference replay (S1): narrative strip + seamless-loop veil. */}
        <LoopCaption />
        <LoopController />
      </div>

      <footer className="px-3 pb-3">
        <Transport />
      </footer>
    </div>
  );
}
