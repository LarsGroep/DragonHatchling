"use client";

/**
 * Dataset switcher + gallery rail (§13, M5). Lists the datasets from the db
 * client (mock or Supabase) and the selected dataset's gallery images; picking
 * a thumbnail loads that pack into the store. Proves end-to-end dataset
 * switching: swapping the dropdown re-drives the whole workbench with a
 * different dataset's packs.
 */
import type { DatasetRow, GalleryImageRow } from "@/src/lib/db/types";
import { useWorkbench } from "@/src/lib/state/store";

export function GalleryStrip({
  datasets,
  activeDataset,
  images,
  onDataset,
}: {
  datasets: DatasetRow[];
  activeDataset: DatasetRow | null;
  images: GalleryImageRow[];
  onDataset: (d: DatasetRow) => void;
}) {
  const imageId = useWorkbench((s) => s.imageId);
  const selectImage = useWorkbench((s) => s.selectImage);

  return (
    <div className="flex items-center gap-3 border-b border-edge bg-panel px-3 py-2">
      <label className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-muted">
        dataset
        <select
          value={activeDataset?.id ?? ""}
          onChange={(e) => {
            const d = datasets.find((x) => x.id === e.target.value);
            if (d) onDataset(d);
          }}
          className="rounded border border-edge bg-panel-hi px-2 py-1 text-xs text-readout focus:border-image focus:outline-none"
        >
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.display_name}
            </option>
          ))}
        </select>
      </label>

      <div className="flex min-w-0 flex-1 items-center gap-2 overflow-x-auto">
        {images.map((img) => {
          const on = img.id === imageId;
          return (
            <button
              key={img.id}
              type="button"
              onClick={() => activeDataset && selectImage(activeDataset.id, img)}
              title={`${img.pred_label} · ${(img.confidence * 100).toFixed(0)}%`}
              className={`group relative h-12 w-12 shrink-0 overflow-hidden rounded border transition-colors ${
                on ? "border-image" : "border-edge hover:border-muted"
              }`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={img.thumb_url}
                alt={img.pred_label}
                className="h-full w-full object-cover"
                style={{ imageRendering: "pixelated" }}
                draggable={false}
              />
              {on ? <span className="absolute inset-0 ring-1 ring-inset ring-image" /> : null}
            </button>
          );
        })}
        {images.length === 0 ? (
          <span className="text-[10px] tracking-widest text-muted">no gallery images</span>
        ) : null}
      </div>
    </div>
  );
}
