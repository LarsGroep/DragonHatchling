import { TopBar } from "@/components/TopBar";
import { TransportBar } from "@/components/TransportBar";
import { WorkbenchPanel } from "@/components/WorkbenchPanel";

/**
 * The workbench shell (§1, §11–§13). A static four-pane layout of the four
 * synchronized spaces plus a transport bar. No WebGL and no data fetching at
 * M0 — every panel is a labeled placeholder that later milestones fill in.
 */
export default function Home() {
  return (
    <div className="flex h-screen flex-col">
      <TopBar />

      <main className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[repeat(4,minmax(180px,1fr))] gap-3 p-3 lg:grid-cols-2 lg:grid-rows-2">
        <WorkbenchPanel
          title="IMAGE SPACE"
          accent="image"
          milestone="M5"
          hint="Original image, patch grid, top-5 class bars, and switchable per-method saliency overlay. The trust anchor."
        />
        <WorkbenchPanel
          title="GAUSSIAN FEATURE FIELD"
          accent="gauss"
          milestone="M6"
          hint="The flagship lens: 197 anisotropic Gaussians whose opacity and glow show importance diffusing across layers."
        />
        <WorkbenchPanel
          title="INTERACTION GRAPH"
          accent="graph"
          milestone="M7"
          hint="Tokens as nodes, top-k attention as edges; per-layer and unrolled all-layers modes behind one WebGL renderer."
        />
        <WorkbenchPanel
          title="LATENT EMBEDDINGS"
          accent="latent"
          milestone="M7"
          hint="Dataset-level UMAP/PCA/t-SNE landscape with the current image's per-layer CLS trajectory as a comet trail."
        />
      </main>

      <footer className="px-3 pb-3">
        <TransportBar />
      </footer>
    </div>
  );
}
