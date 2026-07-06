/**
 * Read-model row types mirroring the Supabase Postgres schema (§15). The web
 * app reads these via the anon key (or from the bundled mock in mock mode).
 */

export interface DatasetRow {
  id: string;
  name: string;
  display_name: string;
  num_classes: number;
  class_names: string[];
  /** Default model for this dataset (models table). */
  model_id: string;
  arch: string;
}

export interface GalleryImageRow {
  id: string;
  dataset_id: string;
  model_id: string;
  class_label: string;
  pred_label: string;
  confidence: number;
  /** Object key within the `packs` bucket, WITH trailing slash (§15 handoff). */
  pack_prefix: string;
  /** Public thumbnail URL. */
  thumb_url: string;
  tags: string[];
}

export interface ProjectionRow {
  id: string;
  dataset_id: string;
  model_id: string;
  layer: number;
  method: string;
  url: string | null;
  reducer_url: string | null;
}

export interface ConceptDictionaryRow {
  id: string;
  model_id: string;
  layer: number;
  url: string;
  quality: Record<string, unknown> | null;
}
