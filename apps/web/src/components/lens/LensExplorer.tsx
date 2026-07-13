"use client";

/**
 * Malignancy Lens Explorer — the client orchestrator for `/lens`
 * (docs/MALIGNANCY-LENS.md). Loads the built-in demo fixture and accepts a real
 * `lens_<dataset>.json` by drag-drop (client-side only). One lens toggle
 * (malignancy / category / manifold) reads the selected lesion three honest
 * ways; a high-sensitivity threshold slider drives the malignancy flag. Every
 * readout traces to a measurement and the manifold view refuses when a lesion is
 * out of distribution — the honest gate that also covers phone uploads.
 */
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LensValidationError, parseLensJson, type LensBundle } from "@/src/lib/lens";
import { Panel } from "../umtvit/controls";
import { LensReadout, type LensMode } from "./LensReadout";

const DEMO_URL = "/lens/demo.json";
const LENSES: Array<{ id: LensMode; label: string }> = [
  { id: "malignancy", label: "Malignant vs benign" },
  { id: "category", label: "Category axis" },
  { id: "manifold", label: "Learned manifold" },
];

export function LensExplorer() {
  const [bundle, setBundle] = useState<LensBundle | null>(null);
  const [source, setSource] = useState("demo");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const dragDepth = useRef(0);

  const [mode, setMode] = useState<LensMode>("malignancy");
  const [sel, setSel] = useState(0);
  const [threshold, setThreshold] = useState(0.2); // high-sensitivity default

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch(DEMO_URL);
        if (!res.ok) throw new Error(`could not load demo (${res.status})`);
        const parsed = parseLensJson(await res.text());
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
      const parsed = parseLensJson(text);
      setBundle(parsed);
      setSource(name);
      setError(null);
      setSel(0);
    } catch (e) {
      setError(`${name}: ${e instanceof LensValidationError ? e.message : e instanceof Error ? e.message : String(e)}`);
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
      const f = e.dataTransfer.files?.[0];
      if (f) loadFile(f);
    },
    [loadFile],
  );

  const lesion = bundle?.lesions[Math.min(sel, (bundle?.lesions.length ?? 1) - 1)] ?? null;
  const malCount = useMemo(
    () => (bundle ? Object.values(bundle.taxonomy.malignant).filter(Boolean).length : 0),
    [bundle],
  );

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
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-edge bg-void/90 px-4 py-2.5 backdrop-blur">
        <div className="flex items-baseline gap-3">
          <span className="text-[15px] font-semibold tracking-tight text-signal">Malignancy lens</span>
          <span className="hidden text-[12px] text-muted md:inline">
            three honest readings · not a diagnostic tool
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="hidden font-mono text-[11px] text-muted sm:inline">{source}</span>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="rounded-md border border-edge bg-panel px-2.5 py-1 text-[11px] font-medium text-readout transition-colors hover:border-muted"
          >
            Load bundle…
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
          <Link href="/sgp" className="rounded-md px-2.5 py-1 text-[11px] font-medium text-muted transition-colors hover:text-readout">SGP</Link>
          <Link href="/" className="rounded-md px-2.5 py-1 text-[11px] font-medium text-muted transition-colors hover:text-readout">← ViTreous</Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-[1080px] flex-col gap-3 p-3 md:p-4">
        {/* honest framing banner — the load-bearing disclaimer, not decoration */}
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-[12px] text-amber-900">
          <span className="font-semibold">Educational demonstration, not a medical device.</span>{" "}
          These readings are derived from a dermatoscopy-trained model on HAM10000. They are not a
          diagnosis, are not clinically validated, and do not report cancer <em>stage</em> (Breslow
          depth, nodal/metastatic status — not present in a surface image). No skin-cancer app has
          FDA approval; treat this as an explainability demo.
        </div>

        {error ? (
          <div className="flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-[12px] text-red-700">
            <span className="font-semibold">Invalid bundle —</span>
            <span className="font-mono">{error}</span>
            <button type="button" onClick={() => setError(null)} className="ml-auto text-red-500 hover:text-red-700" aria-label="Dismiss">✕</button>
          </div>
        ) : null}

        {loading && !bundle ? (
          <div className="rounded-xl border border-edge bg-panel p-8 text-center text-[13px] text-muted shadow-soft">Loading demo…</div>
        ) : null}

        {bundle && lesion ? (
          <>
            {/* lens toggle */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex overflow-hidden rounded-lg border border-edge">
                {LENSES.map((l) => (
                  <button
                    key={l.id}
                    type="button"
                    onClick={() => setMode(l.id)}
                    className={`px-3 py-1.5 text-[12px] font-medium transition-colors ${
                      mode === l.id ? "bg-signal text-white" : "bg-void text-muted hover:text-readout"
                    }`}
                  >
                    {l.label}
                  </button>
                ))}
              </div>
              <span className="text-[11px] text-muted">
                {bundle.dataset} · {bundle.class_names.length} classes · {malCount} malignant
              </span>
            </div>

            <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
              {/* lesion picker + image */}
              <Panel title="Lesion" accent="image" subtitle={lesion.id}>
                <div className="flex flex-col gap-3">
                  <div className="mx-auto aspect-square w-full max-w-[220px] overflow-hidden rounded-lg border border-edge">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`data:image/png;base64,${lesion.thumb_png_b64}`} alt="" className="h-full w-full object-cover" />
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {bundle.lesions.map((l, i) => (
                      <button
                        key={l.id}
                        type="button"
                        onClick={() => setSel(i)}
                        className={`h-10 w-10 overflow-hidden rounded-md border transition-all ${
                          i === sel ? "border-image ring-1 ring-image" : "border-edge opacity-70 hover:opacity-100"
                        }`}
                        aria-label={l.id}
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={`data:image/png;base64,${l.thumb_png_b64}`} alt="" className="h-full w-full object-cover" />
                      </button>
                    ))}
                  </div>
                </div>
              </Panel>

              {/* the reading under the active lens */}
              <Panel
                title="Reading"
                accent="evidence"
                subtitle={LENSES.find((l) => l.id === mode)?.label}
                right={
                  mode === "malignancy" ? (
                    <label className="flex items-center gap-1.5 text-[10px] text-muted">
                      sensitivity
                      <input
                        type="range"
                        min={0.05}
                        max={0.5}
                        step={0.01}
                        value={threshold}
                        onChange={(e) => setThreshold(Number(e.target.value))}
                        style={{ accentColor: "#e11d48" }}
                      />
                    </label>
                  ) : null
                }
              >
                <LensReadout bundle={bundle} lesion={lesion} mode={mode} threshold={threshold} />
              </Panel>
            </div>

            <p className="px-1 pb-4 text-[11px] text-muted">
              Malignant group = {Object.entries(bundle.taxonomy.malignant).filter(([, m]) => m).map(([c]) => c).join(", ")}.
              Axes 1 &amp; 2 are deterministic functions of the diagnosis softmax; axis 3 projects the
              CLS feature onto the learned benign↔malignant axis with an out-of-distribution refusal.
              Drop a run&rsquo;s <code className="rounded bg-panel px-1 font-mono">lens_&lt;dataset&gt;.json</code> anywhere to explore it — nothing leaves your browser.
            </p>
          </>
        ) : null}
      </main>

      {dragging ? (
        <div className="pointer-events-none fixed inset-0 z-20 flex items-center justify-center bg-image/10 backdrop-blur-sm">
          <div className="rounded-2xl border-2 border-dashed border-image bg-void/90 px-8 py-6 text-[14px] font-medium text-image shadow-soft">
            Drop lens_&lt;dataset&gt;.json to explore
          </div>
        </div>
      ) : null}
    </div>
  );
}
