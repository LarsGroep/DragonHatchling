/**
 * Instrument header: product mark plus monospaced status readouts. All values
 * are static placeholders at M0 (no data fetching); the dataset switcher and
 * live model/warm status wire up at M5/M8.
 */

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-[10px] uppercase tracking-widest text-muted">
        {label}
      </span>
      <span className="tabular-nums text-xs text-readout">{value}</span>
    </div>
  );
}

export function TopBar() {
  return (
    <header className="flex items-center justify-between border-b border-edge bg-panel-hi px-4 py-2.5">
      <div className="flex items-baseline gap-3">
        <span className="text-sm font-semibold tracking-[0.3em] text-signal">
          VITREOUS
        </span>
        <span className="hidden text-[11px] text-muted sm:inline">
          explainable vision-transformer workbench
        </span>
      </div>

      <div className="flex items-center gap-5">
        <Readout label="model" value="—" />
        <Readout label="dataset" value="—" />
        <Readout label="image" value="—" />
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-muted" />
          <span className="text-[10px] uppercase tracking-widest text-muted">
            no pack loaded
          </span>
        </div>
      </div>
    </header>
  );
}
