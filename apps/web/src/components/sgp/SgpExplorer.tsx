"use client";

/**
 * SGP Explorer — the client orchestrator for the `/sgp` route
 * (docs/SGP-ARCHITECTURE.md). Loads the built-in demo fixture
 * (`/sgp/demo.json`) on mount and accepts a real run's `sgp_<dataset>.json`
 * (exported by `kaggle_umtvit_sgp.ipynb`) by drag-drop or file picker —
 * client-side only, nothing leaves the browser.
 *
 * One selection model, workbench-style (§11): this component owns the
 * fractional depth clock `t`, the active probe, and the hover/pinned neuron;
 * the lattice, the image overlay, and the U-matrix strata all subscribe to the
 * same state and emit into it — no panel talks to another panel directly.
 */
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  SgpValidationError,
  depthActivations,
  migrationCurve,
  parseSgpJson,
  type SgpBundle,
} from "@/src/lib/sgp";
import { Metric, Panel, PlayScrubber } from "../umtvit/controls";
import { BmuReplayPanel } from "./BmuReplayPanel";
import { SomLatticeView } from "./SomLatticeView";
import { UMatrixPanel } from "./UMatrixPanel";

const DEMO_URL = "/sgp/demo.json";

export function SgpExplorer() {
  const [bundle, setBundle] = useState<SgpBundle | null>(null);
  const [source, setSource] = useState<string>("demo");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const dragDepth = useRef(0);

  // ── the one selection model ──────────────────────────────────────────────
  const [t, setT] = useState(0); // fractional depth 0..Z-1
  const [probeIdx, setProbeIdx] = useState(0);
  const [hoverNeuron, setHoverNeuron] = useState<number | null>(null);
  const [pinnedNeuron, setPinnedNeuron] = useState<number | null>(null);

  // ── bundle loading ───────────────────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch(DEMO_URL);
        if (!res.ok) throw new Error(`could not load demo fixture (${res.status})`);
        const parsed = parseSgpJson(await res.text());
        if (alive) {
          setBundle(parsed);
          setSource("demo");
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const loadText = useCallback((text: string, name: string) => {
    try {
      const parsed = parseSgpJson(text);
      setBundle(parsed);
      setSource(name);
      setError(null);
      setProbeIdx(0);
      setT(0);
      setPinnedNeuron(null);
      setHoverNeuron(null);
    } catch (e) {
      const msg =
        e instanceof SgpValidationError ? e.message : e instanceof Error ? e.message : String(e);
      setError(`${name}: ${msg}`);
    }
  }, []);

  const loadFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = () => loadText(String(reader.result ?? ""), file.name);
      reader.onerror = () => setError(`${file.name}: could not read file`);
      reader.readAsText(file);
    },
    [loadText],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) loadFile(file);
    },
    [loadFile],
  );

  // ── derived, memoized per probe ──────────────────────────────────────────
  const probe = bundle?.probes[Math.min(probeIdx, (bundle?.probes.length ?? 1) - 1)] ?? null;
  const activations = useMemo(
    () => (bundle && probe ? depthActivations(probe.bmu, bundle.som.num_neurons) : null),
    [bundle, probe],
  );
  const migration = useMemo(() => (probe ? migrationCurve(probe.bmu) : []), [probe]);

  const Z = bundle?.som.depth_steps ?? 1;
  const depthInt = Math.max(0, Math.min(Z - 1, Math.round(t)));
  const liveCount = bundle ? bundle.som.num_neurons - bundle.som.dead_neurons : 0;

  return (
    <div
      className="min-h-screen bg-void text-readout"
      onDragEnter={(e) => {
        e.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDrop={onDrop}
    >
      {/* header */}
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-edge bg-void/90 px-4 py-2.5 backdrop-blur">
        <div className="flex items-baseline gap-3">
          <span className="text-[15px] font-semibold tracking-tight text-signal">
            SGP — the SOM as a living graph
          </span>
          <span className="hidden text-[12px] text-muted md:inline">
            real lattice coordinates · measured edges · one image&rsquo;s BMU trail
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="hidden font-mono text-[11px] text-muted sm:inline" title="active run">
            {source}
          </span>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="rounded-md border border-edge bg-panel px-2.5 py-1 text-[11px] font-medium text-readout transition-colors hover:border-muted"
          >
            Load run…
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) loadFile(f);
              e.target.value = "";
            }}
          />
          <Link
            href="/umtvit"
            className="rounded-md px-2.5 py-1 text-[11px] font-medium text-muted transition-colors hover:text-readout"
          >
            UMT-ViT
          </Link>
          <Link
            href="/"
            className="rounded-md px-2.5 py-1 text-[11px] font-medium text-muted transition-colors hover:text-readout"
          >
            ← ViTreous
          </Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-[1280px] flex-col gap-3 p-3 md:p-4">
        {error ? (
          <div className="flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-[12px] text-red-700">
            <span className="font-semibold">Invalid bundle —</span>
            <span className="font-mono">{error}</span>
            <button
              type="button"
              onClick={() => setError(null)}
              className="ml-auto text-red-500 hover:text-red-700"
              aria-label="Dismiss"
            >
              ✕
            </button>
          </div>
        ) : null}

        {loading && !bundle ? (
          <div className="rounded-xl border border-edge bg-panel p-8 text-center text-[13px] text-muted shadow-soft">
            Loading demo run…
          </div>
        ) : null}

        {bundle && probe && activations ? (
          <>
            {/* metrics strip */}
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
              <Metric label="dataset" value={bundle.dataset} />
              <Metric
                label="SOM lattice"
                value={bundle.som.grid.join("×")}
                hint={bundle.som.adjacency}
              />
              <Metric
                label="live neurons"
                value={`${liveCount}/${bundle.som.num_neurons}`}
                tone={liveCount / bundle.som.num_neurons > 0.6 ? "evidence" : "warm"}
                hint={`${bundle.som.dead_neurons} dead (shown, not hidden)`}
              />
              <Metric label="lattice edges" value={String(bundle.som.edges.length)} />
              <Metric
                label="communities"
                value={String(bundle.som.communities.k)}
                hint={`${bundle.som.communities.method} · seed ${bundle.som.communities.seed}`}
              />
              <Metric label="probe images" value={String(bundle.probes.length)} />
            </div>

            <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
              {/* flagship lattice */}
              <Panel
                title="SOM lattice — the learned map"
                accent="gauss"
                subtitle="drag to orbit · hover to link · click to pin"
                right={
                  pinnedNeuron !== null ? (
                    <button
                      type="button"
                      onClick={() => setPinnedNeuron(null)}
                      className="rounded-md border border-edge bg-void px-2 py-0.5 text-[10px] text-readout hover:text-image"
                    >
                      unpin n{pinnedNeuron}
                    </button>
                  ) : null
                }
              >
                <div className="flex h-full min-h-[420px] flex-col gap-2 lg:min-h-[520px]">
                  <div className="min-h-0 flex-1">
                    <SomLatticeView
                      som={bundle.som}
                      activations={activations}
                      t={t}
                      hoverNeuron={hoverNeuron}
                      pinnedNeuron={pinnedNeuron}
                      onHover={setHoverNeuron}
                      onPick={(k) => setPinnedNeuron((p) => (p === k ? null : k))}
                    />
                  </div>
                  <PlayScrubber
                    value={t}
                    min={0}
                    max={Z - 1}
                    step={0.01}
                    speed={(Z - 1) / 7}
                    onChange={setT}
                    accent="gauss"
                    label="depth"
                    format={(v) => `z = ${v.toFixed(2)}`}
                  />
                </div>
              </Panel>

              {/* image side of the sync */}
              <div className="flex flex-col gap-3">
                <Panel
                  title="This image on the map"
                  accent="image"
                  subtitle={`probe ${probe.index} · depth z=${depthInt}`}
                >
                  <div className="flex flex-col gap-3">
                    {/* probe strip */}
                    <div className="flex gap-1.5 overflow-x-auto pb-1">
                      {bundle.probes.map((p, i) => (
                        <button
                          key={i}
                          type="button"
                          onClick={() => setProbeIdx(i)}
                          className={`h-11 w-11 shrink-0 overflow-hidden rounded-md border transition-all ${
                            i === probeIdx
                              ? "border-image ring-1 ring-image"
                              : "border-edge opacity-70 hover:opacity-100"
                          }`}
                          aria-label={`probe ${p.index}`}
                        >
                          {p.thumb_png_b64 ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img
                              src={`data:image/png;base64,${p.thumb_png_b64}`}
                              alt=""
                              className="h-full w-full object-cover"
                            />
                          ) : (
                            <span className="block h-full w-full bg-panel" />
                          )}
                        </button>
                      ))}
                    </div>
                    <BmuReplayPanel
                      som={bundle.som}
                      probe={probe}
                      depth={depthInt}
                      hoverNeuron={hoverNeuron}
                      pinnedNeuron={pinnedNeuron}
                      onHoverNeuron={setHoverNeuron}
                      migration={migration}
                    />
                  </div>
                </Panel>

                <Panel title="Cluster structure" accent="latent" subtitle="per depth-layer strata">
                  <UMatrixPanel
                    som={bundle.som}
                    hoverNeuron={hoverNeuron}
                    pinnedNeuron={pinnedNeuron}
                    onHoverNeuron={setHoverNeuron}
                    onPinNeuron={setPinnedNeuron}
                  />
                </Panel>
              </div>
            </div>

            <p className="px-1 pb-4 text-[11px] text-muted">
              Every encoding is measured: positions are the real{" "}
              {bundle.som.grid.join("×")} neuron lattice, edges are literal grid
              neighbours weighted by weight-space similarity, node size is BMU hit
              count, brightness is this image&rsquo;s BMU share at the scrubbed depth.
              Dead neurons stay visible. Drop a run&rsquo;s{" "}
              <code className="rounded bg-panel px-1 font-mono">sgp_&lt;dataset&gt;.json</code>{" "}
              (exported by <span className="font-mono">kaggle_umtvit_sgp.ipynb</span>) anywhere on
              this page — nothing leaves your browser.
            </p>
          </>
        ) : null}
      </main>

      {/* drag overlay */}
      {dragging ? (
        <div className="pointer-events-none fixed inset-0 z-20 flex items-center justify-center bg-gauss/10 backdrop-blur-sm">
          <div className="rounded-2xl border-2 border-dashed border-gauss bg-void/90 px-8 py-6 text-[14px] font-medium text-gauss shadow-soft">
            Drop sgp_&lt;dataset&gt;.json to explore this run
          </div>
        </div>
      ) : null}
    </div>
  );
}
