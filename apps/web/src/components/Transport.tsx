"use client";

/**
 * Transport bar (§12), wired to the store's timeline clock `t` ∈ [0, L]. The
 * play loop itself runs in Workbench (requestAnimationFrame); this component
 * renders the controls and the scrub track. Keyboard (space/arrows/home/end)
 * is bound in Workbench so it works with focus anywhere.
 */
import { useWorkbench } from "@/src/lib/state/store";

function Btn({
  glyph,
  label,
  onClick,
  disabled,
}: {
  glyph: string;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="flex h-8 w-8 items-center justify-center rounded border border-edge bg-panel text-readout transition-colors hover:border-muted hover:text-signal disabled:cursor-not-allowed disabled:opacity-40"
    >
      <span className="text-sm leading-none">{glyph}</span>
    </button>
  );
}

export function Transport() {
  const packIndex = useWorkbench((s) => s.packIndex);
  const t = useWorkbench((s) => s.t);
  const playing = useWorkbench((s) => s.playing);
  const scrub = useWorkbench((s) => s.scrub);
  const seekLayer = useWorkbench((s) => s.seekLayer);
  const stepLayer = useWorkbench((s) => s.stepLayer);
  const togglePlay = useWorkbench((s) => s.togglePlay);

  const L = packIndex?.numLayers ?? 12;
  const ready = !!packIndex;
  const pct = (t / L) * 100;

  return (
    <div className="flex items-center gap-4 rounded-md border border-edge bg-panel-hi px-3 py-2">
      <div className="flex items-center gap-1.5">
        {/* Ambient-loop liveness lamp: lit while the workbench is replaying (S1). */}
        <span
          className={`mr-1 inline-block h-2 w-2 rounded-full ${
            playing ? "animate-pulse bg-latent shadow-glow shadow-latent" : "bg-muted/50"
          }`}
          title={playing ? "Live — replaying continuously" : "Paused — resumes when idle"}
          aria-hidden
        />
        <Btn glyph="⏮" label="First layer" onClick={() => seekLayer(0)} disabled={!ready} />
        <Btn glyph="◀" label="Step back" onClick={() => stepLayer(-1)} disabled={!ready} />
        <Btn
          glyph={playing ? "❚❚" : "▶"}
          label={playing ? "Pause loop" : "Resume loop"}
          onClick={togglePlay}
          disabled={!ready}
        />
        <Btn glyph="▶▶" label="Step forward" onClick={() => stepLayer(1)} disabled={!ready} />
        <Btn glyph="⏭" label="Final layer" onClick={() => seekLayer(L)} disabled={!ready} />
      </div>

      <div className="flex flex-1 items-center gap-2">
        <span className="text-[10px] tracking-widest text-muted">t</span>
        <div className="relative h-4 flex-1">
          <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-panel">
            <div
              className="absolute inset-y-0 left-0 rounded-full bg-image/40"
              style={{ width: `${pct}%` }}
            />
          </div>
          <input
            type="range"
            min={0}
            max={L}
            step={0.01}
            value={t}
            disabled={!ready}
            onChange={(e) => scrub(Number(e.target.value))}
            aria-label="Timeline scrub"
            className="absolute inset-0 w-full cursor-pointer appearance-none bg-transparent accent-image disabled:cursor-not-allowed"
          />
        </div>
        <span className="tabular-nums text-[11px] text-muted">
          {t.toFixed(1)} / {L}
        </span>
      </div>

      <div className="flex items-center gap-2 text-[10px] tracking-widest text-muted">
        <span>LAYER</span>
        <span className="tabular-nums rounded border border-edge bg-panel px-1.5 py-0.5 text-readout">
          {Math.round(t)}
        </span>
      </div>
    </div>
  );
}
