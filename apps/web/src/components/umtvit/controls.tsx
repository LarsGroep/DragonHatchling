"use client";

/**
 * Shared UI primitives for the UMT-ViT Explorer — a card `Panel` matching the
 * app's light instrument aesthetic, a `PlayScrubber` (range slider + play/pause
 * that advances on a requestAnimationFrame clock and loops), and small metric
 * tiles. No external chart/UI dependency: everything renders with Tailwind
 * tokens already in the theme.
 */
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

export type Accent = "latent" | "gauss" | "image" | "warm" | "evidence";

const ACCENT_HEX: Record<Accent, string> = {
  latent: "#8b5cf6",
  gauss: "#0d9488",
  image: "#3b82f6",
  warm: "#f59e0b",
  evidence: "#22c55e",
};

export function Panel({
  title,
  accent,
  subtitle,
  right,
  children,
}: {
  title: string;
  accent: Accent;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex min-w-0 flex-col overflow-hidden rounded-xl border border-edge bg-panel shadow-soft">
      <header className="flex items-center gap-2 border-b border-edge px-3 py-2">
        <span
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={{ backgroundColor: ACCENT_HEX[accent] }}
        />
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-readout">{title}</h2>
        {subtitle ? <span className="truncate text-[11px] text-muted">· {subtitle}</span> : null}
        {right ? <div className="ml-auto flex items-center gap-2">{right}</div> : null}
      </header>
      <div className="min-h-0 flex-1 p-3">{children}</div>
    </section>
  );
}

function PlayIcon({ playing }: { playing: boolean }) {
  return (
    <svg viewBox="0 0 12 12" className="h-3 w-3" aria-hidden>
      {playing ? (
        <>
          <rect x="2" y="1.5" width="2.6" height="9" fill="currentColor" />
          <rect x="7.4" y="1.5" width="2.6" height="9" fill="currentColor" />
        </>
      ) : (
        <path d="M2.5 1.5 L10.5 6 L2.5 10.5 Z" fill="currentColor" />
      )}
    </svg>
  );
}

/**
 * A range slider paired with a play/pause button. When playing, `value`
 * advances by `speed` units/second on a rAF loop, wrapping from `max` back to
 * `min`. Fully controlled: the parent owns `value` and re-renders the view.
 */
export function PlayScrubber({
  value,
  min,
  max,
  step,
  speed,
  onChange,
  format,
  accent = "image",
  label,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  /** units advanced per second while playing. */
  speed: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  accent?: Accent;
  label?: string;
}) {
  const [playing, setPlaying] = useState(false);
  const valueRef = useRef(value);
  valueRef.current = value;

  useEffect(() => {
    if (!playing || max <= min) return;
    let raf = 0;
    let prev = performance.now();
    const tick = (now: number) => {
      const dt = (now - prev) / 1000;
      prev = now;
      let next = valueRef.current + speed * dt;
      if (next >= max) next = min + ((next - min) % (max - min || 1));
      onChange(next);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, min, max, speed]);

  return (
    <div className="flex items-center gap-2.5">
      <button
        type="button"
        onClick={() => setPlaying((p) => !p)}
        aria-pressed={playing}
        aria-label={playing ? "Pause" : "Play"}
        className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-edge bg-void text-readout transition-colors hover:text-image"
        style={{ color: playing ? ACCENT_HEX[accent] : undefined }}
      >
        <PlayIcon playing={playing} />
      </button>
      {label ? (
        <span className="shrink-0 text-[10px] uppercase tracking-wide text-muted">{label}</span>
      ) : null}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          setPlaying(false);
          onChange(Number(e.target.value));
        }}
        className="min-w-0 flex-1"
        style={{ accentColor: ACCENT_HEX[accent] } as CSSProperties}
      />
      <span className="w-16 shrink-0 text-right font-mono text-[11px] tabular-nums text-readout">
        {format ? format(value) : value.toFixed(0)}
      </span>
    </div>
  );
}

/** A compact labelled metric readout tile. */
export function Metric({
  label,
  value,
  hint,
  tone = "readout",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "readout" | "evidence" | "warm" | "muted";
}) {
  const toneCls =
    tone === "evidence"
      ? "text-evidence"
      : tone === "warm"
        ? "text-warm"
        : tone === "muted"
          ? "text-muted"
          : "text-signal";
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-edge bg-void px-3 py-2">
      <span className="text-[10px] uppercase tracking-wide text-muted">{label}</span>
      <span className={`font-mono text-[15px] font-semibold tabular-nums ${toneCls}`}>{value}</span>
      {hint ? <span className="text-[10px] text-muted">{hint}</span> : null}
    </div>
  );
}

export const fmtNum = (v: number | null, digits = 3): string =>
  v === null || Number.isNaN(v) ? "—" : v.toFixed(digits);
