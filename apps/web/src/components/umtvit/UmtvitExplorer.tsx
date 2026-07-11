"use client";

/**
 * UMT-ViT Explorer — the client orchestrator for the `/umtvit` route. Loads the
 * built-in demo fixture (`/umtvit/demo.json`) on mount, and lets the user drop
 * or pick a real run's `umtvit_web.json` (client-side FileReader, no backend).
 * Validation errors are shown inline and the view stays on the previous bundle
 * on failure. This is a separate surface from the ViTreous workbench and shares
 * none of its store/pack code — only the visual language and Tailwind tokens.
 */
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { parseUmtvitJson, UmtvitValidationError, type UmtvitBundle } from "@/src/lib/umtvit";
import { RunHeader } from "./RunHeader";
import { LatentCube } from "./LatentCube";
import { SomPanel } from "./SomPanel";
import { EmbeddingsPanel } from "./EmbeddingsPanel";
import { TrainingCurves } from "./TrainingCurves";
import { HonestyNote } from "./HonestyNote";

const DEMO_URL = "/umtvit/demo.json";

export function UmtvitExplorer() {
  const [bundle, setBundle] = useState<UmtvitBundle | null>(null);
  const [source, setSource] = useState<string>("demo");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const dragDepth = useRef(0);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch(DEMO_URL);
        if (!res.ok) throw new Error(`could not load demo fixture (${res.status})`);
        const text = await res.text();
        const parsed = parseUmtvitJson(text);
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
      const parsed = parseUmtvitJson(text);
      setBundle(parsed);
      setSource(name);
      setError(null);
    } catch (e) {
      // Keep the previous bundle; surface a helpful, field-named message.
      const msg = e instanceof UmtvitValidationError ? e.message : e instanceof Error ? e.message : String(e);
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
          <span className="text-[15px] font-semibold tracking-tight text-signal">UMT-ViT Explorer</span>
          <span className="hidden text-[12px] text-muted md:inline">
            a topographic latent atlas — learned, not physical, depth
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
            href="/"
            className="rounded-md px-2.5 py-1 text-[11px] font-medium text-muted transition-colors hover:text-readout"
          >
            ← ViTreous
          </Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-[1200px] flex-col gap-3 p-3 md:p-4">
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

        {bundle ? (
          <>
            <RunHeader bundle={bundle} />
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <LatentCube bundle={bundle} />
              <SomPanel bundle={bundle} />
              <EmbeddingsPanel bundle={bundle} />
              <HonestyNote bundle={bundle} />
            </div>
            <TrainingCurves bundle={bundle} />
            <p className="px-1 pb-4 text-[11px] text-muted">
              Drop a run&rsquo;s{" "}
              <code className="rounded bg-panel px-1 font-mono">umtvit_web.json</code> anywhere on this
              page (exported by the Kaggle notebook&rsquo;s{" "}
              <span className="font-mono">Export web bundle</span> cell) to explore it here — nothing
              leaves your browser.
            </p>
          </>
        ) : null}
      </main>

      {/* drag overlay */}
      {dragging ? (
        <div className="pointer-events-none fixed inset-0 z-20 flex items-center justify-center bg-image/10 backdrop-blur-sm">
          <div className="rounded-2xl border-2 border-dashed border-image bg-void/90 px-8 py-6 text-[14px] font-medium text-image shadow-soft">
            Drop umtvit_web.json to explore this run
          </div>
        </div>
      ) : null}
    </div>
  );
}
