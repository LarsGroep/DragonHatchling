/**
 * Placeholder transport bar (§12). At M0 the controls are static and disabled;
 * the replay clock, scrubbing, and per-layer stepping wire up at M6. The
 * timeline reads t = 0..L over the ViT's 12 blocks.
 */

const LAYERS = 12;

function TransportButton({ glyph, label }: { glyph: string; label: string }) {
  return (
    <button
      type="button"
      disabled
      aria-label={label}
      title={`${label} (wires up at M6)`}
      className="flex h-8 w-8 items-center justify-center rounded border border-edge bg-panel text-readout/70 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <span className="text-sm leading-none">{glyph}</span>
    </button>
  );
}

export function TransportBar() {
  return (
    <div className="flex items-center gap-4 rounded-md border border-edge bg-panel-hi px-3 py-2">
      <div className="flex items-center gap-1.5">
        <TransportButton glyph="⏮" label="Step to first layer" />
        <TransportButton glyph="◀" label="Step back one layer" />
        <TransportButton glyph="▶" label="Play" />
        <TransportButton glyph="▶▶" label="Step forward one layer" />
        <TransportButton glyph="⏭" label="Step to final layer" />
      </div>

      {/* Layer track — visual only at M0. */}
      <div className="flex flex-1 items-center gap-2">
        <span className="text-[10px] tracking-widest text-muted">t</span>
        <div className="relative h-1.5 flex-1 rounded-full bg-panel">
          <div className="absolute inset-y-0 left-0 w-0 rounded-full bg-signal/30" />
          <div className="absolute left-0 top-1/2 h-3 w-3 -translate-y-1/2 rounded-full border border-edge bg-signal/60" />
        </div>
        <span className="tabular-nums text-[11px] text-muted">
          0 / {LAYERS}
        </span>
      </div>

      <div className="flex items-center gap-2 text-[10px] tracking-widest text-muted">
        <span>SPEED</span>
        <span className="rounded border border-edge bg-panel px-1.5 py-0.5 text-readout/70">
          1.0×
        </span>
      </div>
    </div>
  );
}
