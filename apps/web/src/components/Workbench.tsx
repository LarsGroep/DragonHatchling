"use client";

/**
 * Workbench — the client orchestrator, redesigned brain-first (UX-VISION-2).
 *
 * Layout inversion: the top ~65% is the living Brain (full width); the bottom
 * strip is three cards — [input image + saliency + method chips] · [Gaussian
 * sensory field] · [concepts + prediction]. A slim transport bar sits beneath
 * the strip; the header is minimal. The ambient loop machinery (store.playing/t/
 * loopStage, LoopController, schedule) is unchanged — it is the replay engine;
 * this component only re-choreographs where its output is drawn.
 *
 * Owns: initial data load (datasets → first gallery → first pack) and global
 * keyboard transport. The per-frame replay clock lives in <LoopController/>.
 */
import { useCallback, useEffect, useState, type ReactNode } from "react";
import type { CSSProperties } from "react";
import type { DatasetRow, GalleryImageRow } from "@/src/lib/db/types";
import { getDb } from "@/src/lib/db/client";
import { useWorkbench } from "@/src/lib/state/store";
import { LOOP_STAGES, type FeaturePane } from "@/src/lib/loop/schedule";
import { WorkbenchHeader } from "./WorkbenchHeader";
import { GalleryStrip } from "./GalleryStrip";
import { Transport } from "./Transport";
import { LoopController } from "./LoopController";
import { LoopCaption } from "./LoopCaption";
import { ImageSpaceView } from "./views/ImageSpace";
import { GaussianFieldView } from "./views/GaussianField";
import { BrainView } from "./views/BrainView";
import { ConceptsPanel } from "./views/ConceptsPanel";

type Accent = "image" | "gauss" | "latent";

const ACCENT_DOT: Record<Accent, string> = {
  image: "bg-image",
  gauss: "bg-gauss",
  latent: "bg-latent",
};
const ACCENT_HEX: Record<Accent, string> = {
  image: "#3b82f6",
  gauss: "#0d9488",
  latent: "#8b5cf6",
};

function StripCard({
  title,
  accent,
  featured,
  children,
}: {
  title: string;
  accent: Accent;
  featured?: boolean;
  children: ReactNode;
}) {
  return (
    <section
      className={`flex min-h-0 min-w-0 flex-col overflow-hidden rounded-xl border border-edge bg-panel shadow-soft ${
        featured ? "pane-featured" : ""
      }`}
      style={featured ? ({ "--vig": ACCENT_HEX[accent] } as CSSProperties) : undefined}
      aria-label={title}
    >
      <header className="flex items-center gap-2 border-b border-edge px-3 py-1.5">
        <span className={`h-1.5 w-1.5 rounded-full ${ACCENT_DOT[accent]}`} />
        <h2 className="text-[11px] font-semibold tracking-wide text-readout">{title}</h2>
      </header>
      <div className="relative min-h-0 flex-1 p-2.5">{children}</div>
    </section>
  );
}

export function Workbench() {
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [activeDataset, setActiveDataset] = useState<DatasetRow | null>(null);
  const [images, setImages] = useState<GalleryImageRow[]>([]);
  const [bootError, setBootError] = useState<string | null>(null);

  const selectImage = useWorkbench((s) => s.selectImage);
  const mode = useWorkbench((s) => s.mode);
  const loopStage = useWorkbench((s) => s.loopStage);
  const feature: FeaturePane = LOOP_STAGES[Math.min(loopStage, LOOP_STAGES.length - 1)].feature;

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

  const plain = mode === "plain";

  return (
    <div className="flex h-screen flex-col bg-void text-readout">
      <WorkbenchHeader datasetName={activeDataset?.display_name} />
      <GalleryStrip
        datasets={datasets}
        activeDataset={activeDataset}
        images={images}
        onDataset={onDataset}
      />

      {bootError ? (
        <div className="border-b border-red-200 bg-red-50 px-4 py-1.5 text-[11px] text-red-600">
          data load failed: {bootError}
        </div>
      ) : null}

      <main className="flex min-h-0 flex-1 flex-col gap-2 p-2">
        {/* THE BRAIN — the centerpiece (~65%). */}
        <section className="relative flex min-h-0 flex-[1.75_1_0%] flex-col overflow-hidden rounded-xl border border-edge bg-panel shadow-soft">
          <LoopCaption />
          <div className="relative min-h-0 flex-1">
            <BrainView />
          </div>
          {/* Seamless-loop fade veil (resets t 12→0 behind a soft white fade). */}
          <LoopController />
        </section>

        {/* BOTTOM STRIP — three supporting cards. */}
        <section className="grid min-h-0 flex-[1_1_0%] grid-cols-1 gap-2 md:grid-cols-3">
          <StripCard
            title={plain ? "The photo & its evidence" : "Image · saliency"}
            accent="image"
            featured={feature === "image"}
          >
            <ImageSpaceView classNames={activeDataset?.class_names} compact />
          </StripCard>
          <StripCard
            title={plain ? "What the model senses" : "Gaussian feature field"}
            accent="gauss"
            featured={feature === "gauss"}
          >
            <GaussianFieldView />
          </StripCard>
          <StripCard
            title={plain ? "Concepts & answer" : "Concepts · prediction"}
            accent="latent"
            featured={feature === "latent"}
          >
            <ConceptsPanel classNames={activeDataset?.class_names} />
          </StripCard>
        </section>
      </main>

      <footer className="px-2 pb-2">
        <Transport />
      </footer>
    </div>
  );
}
