"""Generate kaggle/ham10000_live.ipynb — the single end-to-end live notebook.

Run: python kaggle/_build_ham10000_nb.py
Keeps the notebook JSON valid (nbformat 4) and the code cells in one place so
they can be reviewed as plain Python. Verified against the real vitreous API.
"""
import json
from pathlib import Path

MD = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
CODE = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(MD("""# ViTreous — HAM10000 live pipeline

One **Run All** trains ViT-S/16 on HAM10000, builds Explanation Packs for a
gallery, fits latent projections, (optionally) trains the concept SAE, and
publishes everything to **Supabase** so the deployed workbench goes live.

### Before you run (once)
1. **Attach the dataset**: *Add Input* → search `kmader/skin-cancer-mnist-ham10000` → Add.
2. **Enable Internet + GPU** (T4) in notebook settings.
3. **Add secrets** (*Add-ons → Secrets*): `SUPABASE_SERVICE_KEY` = your service-role key.
   (The project URL + anon key are public and set below.)
4. **Create the tables once**: open the Supabase SQL editor and run
   `supabase/migrations/0001_init.sql` from the repo (idempotent). This notebook
   creates the storage bucket automatically but does **not** run DDL.

Then *Run All*. When it finishes, set the Vercel env vars printed at the end and redeploy.
"""))

cells.append(CODE("""
# ─── Knobs ──────────────────────────────────────────────────────────────
DATASET   = "ham10000"          # this notebook is wired for HAM10000
MODEL     = "vit_s16"
SEED      = 1234
EPOCHS    = 6                   # modest fine-tune for a live demo (raise for accuracy)
BATCH     = 64
LR        = 3e-4
N_GALLERY = 60                  # curated demo images published to the workbench
PROJECTION_LAYERS = (0, 3, 6, 9, 12)
DO_CONCEPTS = True             # train the SAE concept tier (falls back to k-means)
SAE_MAX_IMAGES = 1500          # token-bank size for the SAE ≈ this × 197

# Public Supabase coordinates (safe to commit — anon key is public by design).
SUPABASE_URL = "https://xjsnvobuulfkiibxkbqu.supabase.co"
SUPABASE_ANON = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZ"
                 "iI6Inhqc252b2J1dWxma2lpYnhrYnF1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM0"
                 "MDM0NTksImV4cCI6MjA5ODk3OTQ1OX0.0-wHoqghosePf2bZNAAy84q91kE-dyrV6gKqnf20Cos")
DATA_ROOT = "/kaggle/input/skin-cancer-mnist-ham10000"
OUTPUT_DIR = "/kaggle/working"
"""))

cells.append(CODE("""
# ─── Install the vitreous core package (single code path with the app) ──
import subprocess, sys
REPO   = "https://github.com/LarsGroep/DragonHatchling.git"
BRANCH = "claude/explainable-vit-research-qadg2i"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"git+{REPO}@{BRANCH}#subdirectory=packages/core[ml]",
                "supabase"], check=True)
"""))

cells.append(CODE("""
# ─── Secrets → env (service key stays in Kaggle Secrets, never in the repo) ──
import os
try:
    from kaggle_secrets import UserSecretsClient
    SERVICE_KEY = UserSecretsClient().get_secret("SUPABASE_SERVICE_KEY")
except Exception:
    SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
assert SERVICE_KEY, "Set the SUPABASE_SERVICE_KEY secret (Add-ons → Secrets)."
os.environ.update(
    SUPABASE_URL=SUPABASE_URL,
    SUPABASE_SERVICE_KEY=SERVICE_KEY,
    VITREOUS_STORAGE="supabase",
    VITREOUS_BUCKET="packs",
)
"""))

cells.append(CODE("""
# ─── Supabase client + public storage bucket (idempotent) ──────────────
from supabase import create_client
sb = create_client(SUPABASE_URL, SERVICE_KEY)
try:
    sb.storage.create_bucket("packs", options={"public": True})
    print("created public bucket 'packs'")
except Exception as e:
    print("bucket 'packs' ready (", str(e)[:60], ")")
"""))

cells.append(CODE("""
# ─── Model + data from the adapter (everything derives from DATASET) ────
import torch, numpy as np
from vitreous.data import get_dataset, list_datasets
from vitreous.models import load_model
print("registered:", list_datasets())
adapter = get_dataset(DATASET)()
spec = adapter.spec
print(spec.display_name, "·", spec.num_classes, "classes")

device = "cuda" if torch.cuda.is_available() else "cpu"
lm = load_model(MODEL, spec, pretrained=True, num_classes=spec.num_classes)
model = lm.module.to(device)
"""))

cells.append(CODE("""
# ─── Fine-tune (leak-free lesion-grouped splits come from the adapter) ──
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class DS(Dataset):
    def __init__(self, samples, t): self.s, self.t = list(samples), t
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        img = Image.open(s.image).convert("RGB")
        return self.t(img), s.label

train_dl = DataLoader(DS(adapter.load(DATA_ROOT, "train"), adapter.augment()),
                      batch_size=BATCH, shuffle=True, num_workers=2)
val_dl   = DataLoader(DS(adapter.load(DATA_ROOT, "val"), adapter.preprocess()),
                      batch_size=BATCH, num_workers=2)

torch.manual_seed(SEED)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
lossf = torch.nn.CrossEntropyLoss()
val_acc = 0.0
for ep in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in val_dl:
            c += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); t += len(y)
    val_acc = c / max(t, 1); print(f"epoch {ep}: val_acc={val_acc:.4f}")

ckpt = f"{OUTPUT_DIR}/{MODEL}_{DATASET}.pt"
torch.save(model.state_dict(), ckpt)
model.eval()
"""))

cells.append(CODE("""
# ─── (Optional) concept SAE on layer-9 activations, with k-means fallback ──
concept_spec = None
concept_quality = {}
if DO_CONCEPTS:
    from vitreous.instrument import Instrumenter
    from vitreous.concepts import (train_sae, SAEConceptProvider, KMeansConceptProvider,
                                   build_concept_dictionary, quality_gate, ExemplarRef,
                                   ConceptPackSpec, DEFAULT_LAYER)
    from sklearn.cluster import KMeans
    inst = Instrumenter(model)
    bank, refs = [], []
    train_samples = adapter.load(DATA_ROOT, "train")[:SAE_MAX_IMAGES]
    pre = adapter.preprocess()
    for s in train_samples:
        x = pre(Image.open(s.image).convert("RGB")).unsqueeze(0).to(device)
        tok = inst.capture(x).tokens[DEFAULT_LAYER][0].cpu().numpy()   # [197,384]
        for ti in range(1, tok.shape[0]):                              # skip CLS
            bank.append(tok[ti])
            refs.append(ExemplarRef(image_id=s.image_id, token_idx=ti,
                                    activation=float(np.linalg.norm(tok[ti])),
                                    class_label=spec.class_names[s.label]))
    acts = np.stack(bank).astype("float32")
    print("token bank:", acts.shape)
    try:
        sae, stats = train_sae(acts, epochs=40, seed=SEED, device=device)
        provider = SAEConceptProvider(sae, layer=DEFAULT_LAYER)
        dic = build_concept_dictionary(provider, acts, refs, num_classes=spec.num_classes,
                                       model=MODEL, dataset=DATASET, layer=DEFAULT_LAYER)
        rep = quality_gate(stats, dic)
        concept_quality = {"dead_rate": rep.dead_rate, "coherence": rep.exemplar_coherence,
                           "duplicate_rate": rep.duplicate_rate, "use_sae": rep.use_sae}
        if not rep.use_sae:
            raise RuntimeError("quality gate failed -> k-means")
    except Exception as e:
        print("SAE fallback:", str(e)[:80])
        km = KMeans(n_clusters=256, random_state=SEED, n_init=4).fit(acts)
        provider = KMeansConceptProvider(km.cluster_centers_.astype("float32"), layer=DEFAULT_LAYER)
        dic = build_concept_dictionary(provider, acts, refs, num_classes=spec.num_classes,
                                       model=MODEL, dataset=DATASET, layer=DEFAULT_LAYER)
        concept_quality = {"provider": "kmeans"}
    concept_spec = ConceptPackSpec(provider=provider, dictionary_id=f"{MODEL}_{DATASET}_L{DEFAULT_LAYER}",
                                   layer=DEFAULT_LAYER)
    print("concepts ready:", concept_quality)
"""))

cells.append(CODE("""
# ─── Insert dataset + model rows (service key bypasses RLS) ─────────────
import dataclasses
spec_json = dataclasses.asdict(spec) if dataclasses.is_dataclass(spec) else {
    "display_name": spec.display_name, "num_classes": spec.num_classes,
    "class_names": list(spec.class_names), "image_size": spec.image_size}
ds_row = sb.table("datasets").upsert(
    {"name": DATASET, "display_name": spec.display_name, "spec": spec_json},
    on_conflict="name").execute().data[0]
md_row = sb.table("models").insert(
    {"dataset_id": ds_row["id"], "arch": lm.spec.arch,
     "metrics": {"val_acc": val_acc, "epochs": EPOCHS}}).execute().data[0]
DATASET_ID, MODEL_ID = ds_row["id"], md_row["id"]
print("dataset", DATASET_ID, "model", MODEL_ID)
"""))

cells.append(CODE("""
# ─── Build + publish one Explanation Pack per gallery image ────────────
from vitreous.packs import build_pack
from vitreous.storage import get_storage
import tempfile, os

storage = get_storage("supabase")           # bucket 'packs', creds from env
gallery = adapter.gallery(DATA_ROOT, n=N_GALLERY)
pre = adapter.preprocess()
gallery_rows = []

for s in gallery:
    img = Image.open(s.image).convert("RGB")
    x = pre(img).unsqueeze(0).to(device)
    image_id = s.image_id
    with tempfile.TemporaryDirectory() as td:
        pack_dir = os.path.join(td, image_id)
        build_pack(lm, x, {"id": image_id, "source": "gallery",
                           "width": img.width, "height": img.height},
                   spec, pack_dir, seed=SEED, concepts=concept_spec,
                   model_info=lm.to_model_info_kwargs())
        prefix = f"{DATASET}/{image_id}/"
        storage.put_pack(pack_dir, prefix)                     # -> packs/{ds}/{id}/*
        # thumbnail
        thumb = img.copy(); thumb.thumbnail((96, 96))
        tp = os.path.join(td, "thumb.png"); thumb.save(tp)
        thumb_url = storage.put_file(tp, f"thumbs/{DATASET}/{image_id}.png")
        # read prediction back from the manifest for the row
        import json as _j
        pred = _j.load(open(os.path.join(pack_dir, "manifest.json")))["prediction"]
        dx = s.meta.get("dx")
        gallery_rows.append({
            "dataset_id": DATASET_ID, "model_id": MODEL_ID,
            "class_label": spec.class_names[s.label], "pred_label": pred["label"],
            "confidence": float(pred["confidence"]), "pack_prefix": prefix,
            "thumb_url": thumb_url, "tags": [dx] if dx else []})
    print("published", image_id)

sb.table("gallery_images").insert(gallery_rows).execute()
print("inserted", len(gallery_rows), "gallery rows")
"""))

cells.append(CODE("""
# ─── Dataset-level latent projections (§10) ────────────────────────────
from vitreous.projections import build_projection_artifacts
from vitreous.instrument import Instrumenter
inst = Instrumenter(model)

# collect per-layer CLS across the gallery for the projection basis
per_layer = {L: [] for L in PROJECTION_LAYERS}
for s in gallery:
    x = pre(Image.open(s.image).convert("RGB")).unsqueeze(0).to(device)
    tks = inst.capture(x).tokens                    # [13,197,384]
    for L in PROJECTION_LAYERS:
        per_layer[L].append(tks[L][0].cpu().numpy())  # CLS token 0

proj_rows = []
with tempfile.TemporaryDirectory() as td:
    for L in PROJECTION_LAYERS:
        cls = np.stack(per_layer[L]).astype("float32")
        arts = build_projection_artifacts(td, cls, layer=L, dataset=DATASET, model=MODEL, seed=SEED)
        for method, m in arts["methods"].items():
            url = storage.put_file(os.path.join(td, m["coords_file"]),
                                   f"projections/{DATASET}/{MODEL}/{m['coords_file']}")
            red = None
            if m.get("reducer_file"):
                red = storage.put_file(os.path.join(td, m["reducer_file"]),
                                       f"projections/{DATASET}/{MODEL}/{m['reducer_file']}")
            proj_rows.append({"dataset_id": DATASET_ID, "model_id": MODEL_ID, "layer": L,
                              "method": method, "url": url, "reducer_url": red})
if proj_rows:
    sb.table("projections").upsert(proj_rows,
        on_conflict="dataset_id,model_id,layer,method").execute()
print("inserted", len(proj_rows), "projection rows")
"""))

cells.append(CODE("""
# ─── Concept dictionary row (if the SAE/k-means tier ran) ──────────────
if concept_spec is not None:
    import json as _j, tempfile, os
    from vitreous.concepts import DEFAULT_LAYER
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "dictionary.json")
        _j.dump(dic.to_json() if hasattr(dic, "to_json") else {}, open(p, "w"))
        url = storage.put_file(p, f"concepts/{DATASET}/{MODEL}_L{DEFAULT_LAYER}.json")
    sb.table("concept_dictionaries").insert(
        {"model_id": MODEL_ID, "layer": DEFAULT_LAYER, "url": url,
         "quality": concept_quality}).execute()
    print("concept dictionary published")
"""))

cells.append(CODE("""
# ─── Done — set these on Vercel and redeploy ───────────────────────────
print("HAM10000 is live. Set on the Vercel project (apps/web) and redeploy:")
print("  NEXT_PUBLIC_SUPABASE_URL      =", SUPABASE_URL)
print("  NEXT_PUBLIC_SUPABASE_ANON_KEY =", SUPABASE_ANON[:24], "…(the anon key above)")
print("Then the dataset switcher will list HAM10000 with", len(gallery_rows), "gallery images.")
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "vitreous": {"dataset": "ham10000",
                                "symbols": ["get_dataset", "load_model", "Instrumenter",
                                            "build_pack", "build_projection_artifacts",
                                            "get_storage", "train_sae", "quality_gate",
                                            "build_concept_dictionary", "ConceptPackSpec"]}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "ham10000_live.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
