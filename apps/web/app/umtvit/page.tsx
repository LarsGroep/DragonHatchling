import type { Metadata } from "next";
import { UmtvitExplorer } from "@/src/components/umtvit/UmtvitExplorer";

/**
 * `/umtvit` — the UMT-ViT Explorer, a separate surface from the ViTreous
 * workbench for exploring a UMT-ViT run's `umtvit_web.json` (latent cube, SOM,
 * embeddings, training curves, Z-axis honesty). All behavior lives in the
 * client `UmtvitExplorer`; this route is a thin shell.
 */
export const metadata: Metadata = {
  title: "UMT-ViT Explorer — a topographic latent atlas",
  description:
    "Explore a UMT-ViT run: the latent voxel cube, its self-organizing map, embedding formation, and whether learned-depth scale ordering emerged.",
};

export default function UmtvitPage() {
  return <UmtvitExplorer />;
}
