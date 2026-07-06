import type { ReactNode } from "react";

export type PanelAccent = "image" | "gauss" | "graph" | "latent";

const ACCENT_TEXT: Record<PanelAccent, string> = {
  image: "text-image",
  gauss: "text-gauss",
  graph: "text-graph",
  latent: "text-latent",
};

const ACCENT_DOT: Record<PanelAccent, string> = {
  image: "bg-image",
  gauss: "bg-gauss",
  graph: "bg-graph",
  latent: "bg-latent",
};

interface WorkbenchPanelProps {
  /** Short view name, e.g. "IMAGE SPACE". */
  title: string;
  /** Which of the four synchronized spaces this is (§1). */
  accent: PanelAccent;
  /** Milestone that lights this panel up (wayfinding for the scaffold). */
  milestone: string;
  /** One-line description of what will render here. */
  hint: string;
  children?: ReactNode;
}

/**
 * A single pane of the four-pane workbench. At M0 it is a labeled placeholder:
 * accent-coded header, a monospaced readout stub, and an empty stage where the
 * WebGL view mounts in later milestones. No data, no WebGL.
 */
export function WorkbenchPanel({
  title,
  accent,
  milestone,
  hint,
  children,
}: WorkbenchPanelProps) {
  return (
    <section
      className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-md border border-edge bg-panel"
      aria-label={title}
    >
      <header className="flex items-center justify-between border-b border-edge bg-panel-hi px-3 py-2">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${ACCENT_DOT[accent]}`} />
          <h2 className={`text-xs font-medium tracking-[0.18em] ${ACCENT_TEXT[accent]}`}>
            {title}
          </h2>
        </div>
        <span className="text-[10px] uppercase tracking-widest text-muted">
          {milestone}
        </span>
      </header>

      <div className="relative flex flex-1 items-center justify-center p-4">
        {children ?? (
          <div className="pointer-events-none flex flex-col items-center gap-2 text-center">
            <span className="text-[11px] tracking-widest text-muted">
              awaiting pack
            </span>
            <span className="max-w-[28ch] text-[11px] leading-relaxed text-muted/70">
              {hint}
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
