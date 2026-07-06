/**
 * Data hook for Image Space: loads the current method's attribution and (once
 * per pack) faithfulness.json from the store's PackClient. Keyed off the loaded
 * manifest + method so it re-fetches only when those change; cancels stale
 * loads on rapid switching.
 */
import { useEffect, useState } from "react";
import type { FaithfulnessJson, LoadedAttribution } from "@/src/lib/pack/types";
import { useWorkbench } from "@/src/lib/state/store";

export function useImageSpaceData() {
  const client = useWorkbench((s) => s.client);
  const manifest = useWorkbench((s) => s.manifest);
  const method = useWorkbench((s) => s.method);

  const [attribution, setAttribution] = useState<LoadedAttribution | null>(null);
  const [faithfulness, setFaithfulness] = useState<FaithfulnessJson | null>(null);
  const [attrError, setAttrError] = useState<string | null>(null);

  // attribution — refetch on pack or method change
  useEffect(() => {
    let alive = true;
    setAttribution(null);
    setAttrError(null);
    if (!client || !manifest) return;
    client
      .loadAttribution(method, manifest)
      .then((a) => alive && setAttribution(a))
      .catch((e) => alive && setAttrError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [client, manifest, method]);

  // faithfulness — once per pack
  useEffect(() => {
    let alive = true;
    setFaithfulness(null);
    if (!client || !manifest) return;
    if (!("faithfulness.json" in manifest.assets)) return;
    client
      .loadFaithfulness()
      .then((f) => alive && setFaithfulness(f))
      .catch(() => alive && setFaithfulness(null));
    return () => {
      alive = false;
    };
  }, [client, manifest]);

  return { attribution, faithfulness, attrError };
}
