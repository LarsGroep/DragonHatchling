/**
 * VitreousDb — thin, typed, read-only data client (§15). Exposes the four reads
 * the workbench needs; backed either by Supabase (anon key) or the bundled mock
 * fixture. Callers never branch on mode — they just await these methods.
 */
import { getStorageConfig, type StorageConfig } from "./config";
import type {
  ConceptDictionaryRow,
  DatasetRow,
  GalleryImageRow,
  ProjectionRow,
} from "./types";

interface MockDoc {
  storage: { bucket: string };
  datasets: DatasetRow[];
  gallery: Record<string, GalleryImageRow[]>;
  projections: ProjectionRow[];
}

export interface VitreousDb {
  readonly config: StorageConfig;
  listDatasets(): Promise<DatasetRow[]>;
  listGalleryImages(datasetId: string): Promise<GalleryImageRow[]>;
  getProjections(datasetId: string, modelId: string): Promise<ProjectionRow[]>;
  getConceptDictionary(modelId: string, layer: number): Promise<ConceptDictionaryRow | null>;
}

// --------------------------------------------------------------------------- //
// Mock implementation (bundled fixture — zero backend)
// --------------------------------------------------------------------------- //

class MockDb implements VitreousDb {
  readonly config: StorageConfig;
  private cache: Promise<MockDoc> | null = null;

  constructor(config: StorageConfig) {
    this.config = config;
  }

  private load(): Promise<MockDoc> {
    if (!this.cache) {
      this.cache = fetch(`${this.config.mockBase}/datasets.json`).then((r) => {
        if (!r.ok) throw new Error(`mock datasets.json: ${r.status}`);
        return r.json() as Promise<MockDoc>;
      });
    }
    return this.cache;
  }

  async listDatasets(): Promise<DatasetRow[]> {
    return (await this.load()).datasets;
  }

  async listGalleryImages(datasetId: string): Promise<GalleryImageRow[]> {
    return (await this.load()).gallery[datasetId] ?? [];
  }

  async getProjections(datasetId: string, modelId: string): Promise<ProjectionRow[]> {
    const doc = await this.load();
    return doc.projections.filter(
      (p) => p.dataset_id === datasetId && p.model_id === modelId,
    );
  }

  async getConceptDictionary(): Promise<ConceptDictionaryRow | null> {
    // The mock carries the concept tier inside the pack (concepts.json); there
    // is no separate dictionary object in M5.
    return null;
  }
}

// --------------------------------------------------------------------------- //
// Supabase implementation (anon key, read-only)
// --------------------------------------------------------------------------- //

class SupabaseDb implements VitreousDb {
  readonly config: StorageConfig;
  // Loaded lazily so the mock path never imports @supabase/supabase-js.
  private clientPromise: Promise<import("@supabase/supabase-js").SupabaseClient> | null = null;

  constructor(config: StorageConfig) {
    this.config = config;
  }

  private client() {
    if (!this.clientPromise) {
      this.clientPromise = import("@supabase/supabase-js").then(({ createClient }) =>
        createClient(this.config.supabaseUrl!, this.config.anonKey!, {
          auth: { persistSession: false },
        }),
      );
    }
    return this.clientPromise;
  }

  async listDatasets(): Promise<DatasetRow[]> {
    const sb = await this.client();
    const { data, error } = await sb
      .from("datasets")
      .select("id, name, spec, models(id, arch)")
      .order("name");
    if (error) throw error;
    // Flatten spec jsonb + default model into the read-model row.
    return (data ?? []).map((d: Record<string, unknown>) => {
      const spec = (d.spec ?? {}) as Record<string, unknown>;
      const models = (d.models ?? []) as Array<{ id: string; arch: string }>;
      return {
        id: String(d.id),
        name: String(d.name),
        display_name: String(spec.display_name ?? d.name),
        num_classes: Number(spec.num_classes ?? 0),
        class_names: (spec.class_names as string[]) ?? [],
        model_id: models[0]?.id ?? "",
        arch: models[0]?.arch ?? "",
      } satisfies DatasetRow;
    });
  }

  async listGalleryImages(datasetId: string): Promise<GalleryImageRow[]> {
    const sb = await this.client();
    const { data, error } = await sb
      .from("gallery_images")
      .select("id, dataset_id, model_id, class_label, pred_label, confidence, pack_prefix, thumb_url, tags")
      .eq("dataset_id", datasetId)
      .order("id");
    if (error) throw error;
    return (data ?? []) as GalleryImageRow[];
  }

  async getProjections(datasetId: string, modelId: string): Promise<ProjectionRow[]> {
    const sb = await this.client();
    const { data, error } = await sb
      .from("projections")
      .select("id, dataset_id, model_id, layer, method, url, reducer_url")
      .eq("dataset_id", datasetId)
      .eq("model_id", modelId);
    if (error) throw error;
    return (data ?? []) as ProjectionRow[];
  }

  async getConceptDictionary(
    modelId: string,
    layer: number,
  ): Promise<ConceptDictionaryRow | null> {
    const sb = await this.client();
    const { data, error } = await sb
      .from("concept_dictionaries")
      .select("id, model_id, layer, url, quality")
      .eq("model_id", modelId)
      .eq("layer", layer)
      .limit(1)
      .maybeSingle();
    if (error) throw error;
    return (data as ConceptDictionaryRow) ?? null;
  }
}

// --------------------------------------------------------------------------- //

let singleton: VitreousDb | null = null;

/** Get the process-wide db client (mock or Supabase per env). */
export function getDb(): VitreousDb {
  if (!singleton) {
    const config = getStorageConfig();
    singleton = config.mode === "mock" ? new MockDb(config) : new SupabaseDb(config);
  }
  return singleton;
}
