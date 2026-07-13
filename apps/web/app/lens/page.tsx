import type { Metadata } from "next";
import { LensExplorer } from "@/src/components/lens/LensExplorer";

/**
 * `/lens` — the Malignancy Lens: three honest, label-free readings of a
 * HAM10000 pack (malignant/benign, a benign→in-situ→invasive category axis, and
 * an unsupervised manifold position with an out-of-distribution refusal).
 * Explicitly an explainability demo, NOT a diagnostic tool. Contract:
 * docs/MALIGNANCY-LENS.md. All behavior lives in the client `LensExplorer`.
 */
export const metadata: Metadata = {
  title: "Malignancy lens — an honest clinical reading",
  description:
    "Three label-free readings of a skin-lesion model: malignant vs benign, a benign→in-situ→invasive category axis, and position on the learned manifold with an out-of-distribution refusal. Educational, not diagnostic.",
};

export default function LensPage() {
  return <LensExplorer />;
}
