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
import { useCallback, useEffect, useRef, useState } from "react";
import type { DatasetRow, GalleryImageRow } from "@/src/lib/db/types";
import { getDb } from "@/src/lib/db/client";
import { useWorkbench } from "@/src/lib/state/store";
import { WorkbenchPanel } from "@/components/WorkbenchPanel";
import { WorkbenchHeader } from "./WorkbenchHeader";
import { GalleryStrip } from "./GalleryStrip";
import { Transport } from "./Transport";
import { SyncedPlaceholder } from "./SyncedPlaceholder";
import { ImageSpaceView } from "./views/ImageSpace";
import { GaussianFieldView } from "./views/GaussianField";

/** Timeline layers swept per second at 1× speed. */
const LAYERS_PER_SEC = 2;

export function Workbench() {
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [activeDataset, setActiveDataset] = useState<DatasetRow | null>(null);
  const [images, setImages] = useState<GalleryImageRow[]>([]);
  const [bootError, setBootError] = useState<string | null>(null);

  const selectImage = useWorkbench((s) => s.selectImage);

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

  // -- rAF replay clock ---------------------------------------------------- //
  const raf = useRef<number | null>(null);
  const last = useRef<number>(0);
  useEffect(() => {
    function tick(now: number) {
      const st = useWorkbench.getState();
      const L = st.packIndex?.numLayers ?? 12;
      if (st.playing) {
        const dt = last.current ? (now - last.current) / 1000 : 0;
        let nt = st.t + dt * LAYERS_PER_SEC * st.speed;
        if (nt >= L) nt = 0; // loop
        st.setT(nt);
      }
      last.current = now;
      raf.current = requestAnimationFrame(tick);
    }
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current != null) cancelAnimationFrame(raf.current);
      last.current = 0;
    };
  }, []);

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

      <main className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[repeat(4,minmax(180px,1fr))] gap-3 p-3 lg:grid-cols-2 lg:grid-rows-2">
        <WorkbenchPanel title="IMAGE SPACE" accent="image" milestone="M5" hint="">
          <div className="h-full w-full">
            <ImageSpaceView classNames={activeDataset?.class_names} />
          </div>
        </WorkbenchPanel>

        <WorkbenchPanel title="GAUSSIAN FEATURE FIELD" accent="gauss" milestone="M6" hint="">
          <div className="h-full w-full">
            <GaussianFieldView />
          </div>
        </WorkbenchPanel>

        <WorkbenchPanel title="INTERACTION GRAPH" accent="graph" milestone="M7" hint="">
          <SyncedPlaceholder
            accent="graph"
            emphasis="node"
            hint="Tokens as nodes, top-k attention as edges; per-layer and unrolled all-layers modes."
          />
        </WorkbenchPanel>

        <WorkbenchPanel title="LATENT EMBEDDINGS" accent="latent" milestone="M7" hint="">
          <SyncedPlaceholder
            accent="latent"
            emphasis="point"
            hint="Dataset-level UMAP/PCA/t-SNE landscape with the image's per-layer CLS trajectory."
          />
        </WorkbenchPanel>
      </main>

      <footer className="px-3 pb-3">
        <Transport />
      </footer>
    </div>
  );
}
