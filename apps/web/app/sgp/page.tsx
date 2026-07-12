import type { Metadata } from "next";
import { SgpExplorer } from "@/src/components/sgp/SgpExplorer";

/**
 * `/sgp` — the SGP (SomGraphProvider) Explorer: UMT-ViT's learned 3-D SOM
 * rendered as a native graph (real lattice coordinates, measured U-matrix
 * edges) with a per-image BMU activation replay across encoder depth. All
 * behavior lives in the client `SgpExplorer`; this route is a thin shell.
 * Contract: docs/SGP-ARCHITECTURE.md; producer: kaggle_umtvit_sgp.ipynb.
 */
export const metadata: Metadata = {
  title: "SGP — a learned map, rendered honestly",
  description:
    "Explore a UMT-ViT run's self-organizing map as a living graph: real neuron lattice coordinates, measured cluster boundaries, and a single image's BMU trail across learned depth.",
};

export default function SgpPage() {
  return <SgpExplorer />;
}
