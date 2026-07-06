# vitreous — ViTreous core

The single Python package behind the ViTreous explainable-ViT workbench
(see `docs/ARCHITECTURE.md`). Instrumentation, XAI, Gaussian feature fields,
interaction-graph providers, the concept tier, Explanation Pack tooling, and
storage adapters all live here so Kaggle notebooks and the Hugging Face Space
share one code path.

## Install

```bash
pip install -e packages/core            # M0 runtime: pydantic, numpy, jsonschema
pip install -e "packages/core[ml]"      # adds torch, torchvision, timm (M1+)
pip install -e "packages/core[dev]"     # adds pytest
```

Importing `vitreous` never requires torch/timm — the ML stack is an optional
`[ml]` extra, loaded lazily where model logic lands (M1+).

## Milestone status

M0 ships real interfaces (Protocols/ABCs + dataclasses) with the dataset
**registry fully working** and the Explanation Pack manifest models complete;
compute-bearing methods raise `NotImplementedError` until their milestone
(§16). Run the tests with `pytest packages/core`.
