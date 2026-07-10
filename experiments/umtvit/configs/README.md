# UMT-ViT run configs

Each YAML in this directory fully describes one UMT-ViT run: dataset, backbone,
losses, and training schedule (the schema is `umtvit.config.Config`, the single
source of truth). **Swapping datasets is a config-only operation** — point the
notebook or trainer at a different file here (`shapes.yaml` for the zero-download
CI smoke, `ham10000.yaml`, `eurosat.yaml`, ... as they land in U1) and every
downstream stage derives from it; no model or training code reads dataset
specifics from anywhere else. Load and validate one with
`umtvit.config.load_config("configs/shapes.yaml")`, which returns a structurally
valid `Config` or raises `ConfigError` naming the offending field.
