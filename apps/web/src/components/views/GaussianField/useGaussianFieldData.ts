/**
 * Data hook for the Gaussian Feature Field: loads gaussians.bin ONCE per pack
 * from the store's PackClient (mirrors useImageSpaceData). Reports absence
 * gracefully — a pack may legitimately carry no gaussian asset.
 */
import { useEffect, useState } from "react";
import type { LoadedGaussians } from "@/src/lib/pack/types";
import { useWorkbench } from "@/src/lib/state/store";

export interface GaussianFieldData {
  gaussians: LoadedGaussians | null;
  absent: boolean;
  error: string | null;
}

export function useGaussianFieldData(): GaussianFieldData {
  const client = useWorkbench((s) => s.client);
  const manifest = useWorkbench((s) => s.manifest);

  const [gaussians, setGaussians] = useState<LoadedGaussians | null>(null);
  const [absent, setAbsent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setGaussians(null);
    setError(null);
    setAbsent(false);
    if (!client || !manifest) return;
    if (!("gaussians.bin" in manifest.assets)) {
      setAbsent(true);
      return;
    }
    client
      .loadGaussians(manifest)
      .then((g) => alive && setGaussians(g))
      .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [client, manifest]);

  return { gaussians, absent, error };
}
