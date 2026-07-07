/**
 * The single Zustand selection store (§11). One selection model; no view talks
 * to another view directly — every view is a subscriber (renders highlight) and
 * an emitter (publishes hover/select). The store also owns the loaded pack
 * (manifest + PackClient + PackIndex) and the timeline clock `t` (§12).
 */
import { create } from "zustand";
import type { PackManifest } from "@vitreous/schema";
import { PackClient } from "../pack/PackClient";
import { getDb } from "../db/client";
import { packUrlFor } from "../db/config";
import type { GalleryImageRow } from "../db/types";
import { buildPackIndex, type PackIndex } from "./packIndex";
import type { AttributionMethod, EntityRef } from "./refs";
import { refKey } from "./refs";

export interface WorkbenchState {
  // -- loaded pack -------------------------------------------------------- //
  datasetId: string | null;
  imageId: string | null;
  gallery: GalleryImageRow | null;
  client: PackClient | null;
  manifest: PackManifest | null;
  packIndex: PackIndex | null;
  loading: boolean;
  error: string | null;

  // -- selection model (§11) --------------------------------------------- //
  t: number; // timeline position 0..L, fractional
  hover: EntityRef | null;
  pinned: EntityRef[];
  method: AttributionMethod;

  // -- presentation ------------------------------------------------------- //
  /** Plain (lay, guided) vs Expert (methods, layers). Default plain. */
  mode: "plain" | "expert";

  // -- transport + ambient loop (§12, S1) --------------------------------- //
  playing: boolean; // ambient loop running (auto-started on image select)
  speed: number;
  /** Current narrative stage index (published by the LoopController). */
  loopStage: number;
  /** Wall-clock (ms) of the last user transport interaction; drives idle-resume. */
  lastInteraction: number;

  // -- actions ------------------------------------------------------------ //
  selectImage(datasetId: string, image: GalleryImageRow): Promise<void>;
  setT(t: number): void;
  /** User-driven timeline change (scrub/step): sets t, pauses, marks interaction. */
  scrub(t: number): void;
  setHover(ref: EntityRef | null): void;
  pin(ref: EntityRef): void;
  unpin(ref: EntityRef): void;
  togglePin(ref: EntityRef): void;
  clearPins(): void;
  setMethod(method: AttributionMethod): void;
  setMode(mode: "plain" | "expert"): void;
  toggleMode(): void;
  play(): void;
  pause(): void;
  togglePlay(): void;
  stepLayer(delta: number): void;
  seekLayer(layer: number): void;
  setLoopStage(stage: number): void;
}

/** Effective (integer) attention layer for the current fractional t. */
export function layerForT(t: number, numLayers: number): number {
  return Math.max(0, Math.min(numLayers - 1, Math.round(t)));
}

export const useWorkbench = create<WorkbenchState>((set, get) => ({
  datasetId: null,
  imageId: null,
  gallery: null,
  client: null,
  manifest: null,
  packIndex: null,
  loading: false,
  error: null,

  t: 0,
  hover: null,
  pinned: [],
  method: "chefer",

  mode: "plain",

  playing: false,
  speed: 1,
  loopStage: 0,
  lastInteraction: 0,

  async selectImage(datasetId, image) {
    // Guard against races: only the latest selection may commit.
    const token = `${datasetId}/${image.id}/${Date.now()}`;
    (get() as unknown as { _sel?: string })._sel = token;
    set({
      loading: true,
      error: null,
      datasetId,
      imageId: image.id,
      gallery: image,
      playing: false,
      hover: null,
      pinned: [],
    });
    try {
      const cfg = getDb().config;
      const client = new PackClient(packUrlFor(cfg, image.pack_prefix));
      const manifest = await client.loadManifest();
      const concepts = await client.loadConcepts(manifest);
      const packIndex = buildPackIndex(manifest, concepts);
      if ((get() as unknown as { _sel?: string })._sel !== token) return; // superseded
      // Auto-start the ambient loop: the workbench simply IS alive (S1).
      set({
        client,
        manifest,
        packIndex,
        loading: false,
        t: 0,
        loopStage: 0,
        playing: true,
        lastInteraction: 0,
      });
    } catch (err) {
      if ((get() as unknown as { _sel?: string })._sel !== token) return;
      set({
        loading: false,
        error: err instanceof Error ? err.message : String(err),
        client: null,
        manifest: null,
        packIndex: null,
      });
    }
  },

  setT(t) {
    const L = get().packIndex?.numLayers ?? 12;
    set({ t: Math.max(0, Math.min(L, t)) });
  },

  scrub(t) {
    const L = get().packIndex?.numLayers ?? 12;
    // A scrub is a user interaction: pause the loop and arm the idle-resume timer.
    set({ t: Math.max(0, Math.min(L, t)), playing: false, lastInteraction: Date.now() });
  },

  setHover(ref) {
    set({ hover: ref });
  },

  pin(ref) {
    const key = refKey(ref);
    set((s) => (s.pinned.some((p) => refKey(p) === key) ? s : { pinned: [...s.pinned, ref] }));
  },

  unpin(ref) {
    const key = refKey(ref);
    set((s) => ({ pinned: s.pinned.filter((p) => refKey(p) !== key) }));
  },

  togglePin(ref) {
    const key = refKey(ref);
    set((s) =>
      s.pinned.some((p) => refKey(p) === key)
        ? { pinned: s.pinned.filter((p) => refKey(p) !== key) }
        : { pinned: [...s.pinned, ref] },
    );
  },

  clearPins() {
    set({ pinned: [] });
  },

  setMethod(method) {
    set({ method });
  },

  setMode(mode) {
    set({ mode });
  },
  toggleMode() {
    set((s) => ({ mode: s.mode === "plain" ? "expert" : "plain" }));
  },

  play() {
    set({ playing: true, lastInteraction: 0 });
  },
  pause() {
    set({ playing: false, lastInteraction: Date.now() });
  },
  togglePlay() {
    set((s) => ({ playing: !s.playing, lastInteraction: Date.now() }));
  },

  stepLayer(delta) {
    const L = get().packIndex?.numLayers ?? 12;
    const cur = Math.round(get().t);
    // Stepping is a user interaction: pause + arm idle-resume.
    set({
      t: Math.max(0, Math.min(L, cur + delta)),
      playing: false,
      lastInteraction: Date.now(),
    });
  },

  seekLayer(layer) {
    const L = get().packIndex?.numLayers ?? 12;
    set({
      t: Math.max(0, Math.min(L, layer)),
      playing: false,
      lastInteraction: Date.now(),
    });
  },

  setLoopStage(stage) {
    if (get().loopStage !== stage) set({ loopStage: stage });
  },
}));
