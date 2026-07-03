"""Export a trained classifier as an in-browser inference bundle.

The bundle is what the web app (``webapp/``) consumes to run inference with
onnxruntime-web and light up the Hebbian graph:

* ``model.onnx`` — the classifier with **extra outputs**: alongside
  ``logits`` it emits, per Hebbian-observed layer, the pooled, rectified
  activations of exactly the units the :class:`HebbianFeatureMemory`
  tracked (same subsampling, same order), named ``act_<layer>``.  Unit ``i``
  of ``act_<layer>`` therefore corresponds to graph node ``u:<layer>:<i>``.
* ``manifest.json`` — everything the browser needs: preprocessing constants
  (image size, mean/std), class names, and the mapping from activation
  outputs to graph node ids.

Nothing here is dataset- or backbone-specific; the manifest is derived from
the :class:`DatasetSpec` and the memory, so a dataset swap re-exports
automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from hatchvision.data.base import DatasetSpec
from hatchvision.hebbian.memory import HebbianFeatureMemory


class _WithActivations(nn.Module):
    """Wraps a classifier so tracked activations become graph outputs."""

    def __init__(
        self,
        model: nn.Module,
        layers: Dict[str, nn.Module],
        unit_indices: Dict[str, Optional[torch.Tensor]],
    ) -> None:
        super().__init__()
        self.model = model
        self.layer_names = list(layers)
        self._acts: Dict[str, torch.Tensor] = {}
        self._indices = unit_indices
        self._handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in layers.items()
        ]

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_m, _i, out):
            a = out
            if a.dim() == 4:
                a = a.mean(dim=(2, 3))
            elif a.dim() == 3:
                a = a.mean(dim=1)
            a = torch.relu(a)
            idx = self._indices.get(name)
            if idx is not None:
                a = a.index_select(1, idx)
            self._acts[name] = a
        return hook

    def forward(self, x: torch.Tensor):
        logits = self.model(x)
        return tuple([logits] + [self._acts[n] for n in self.layer_names])


def export_onnx_bundle(
    model: nn.Module,
    memory: HebbianFeatureMemory,
    spec: DatasetSpec,
    out_dir: Union[str, Path],
    graph_file: str = "graph.json",
    model_file: str = "model.onnx",
    manifest_file: str = "manifest.json",
    opset: int = 17,
    extra_meta: Optional[Dict] = None,
) -> Path:
    """Write ``model.onnx`` + ``manifest.json`` into ``out_dir``.

    The IVGraph JSON is exported separately (``export_ivgraph``); pass its
    filename via ``graph_file`` so the manifest can reference it.
    Returns the manifest path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = model.hebbian_layers()
    tracked = {name: layers[name] for name in memory.layer_names if name in layers}
    if not tracked:
        raise ValueError("memory tracks no layers present on the model")
    indices = {
        name: memory.stats[name].unit_index for name in tracked
    }

    wrapper = _WithActivations(model, tracked, indices)
    wrapper.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(1, spec.in_channels, spec.image_size, spec.image_size, device=device)
    output_names = ["logits"] + [f"act_{n}" for n in tracked]
    dynamic_axes = {"images": {0: "batch"}}
    dynamic_axes.update({o: {0: "batch"} for o in output_names})

    onnx_path = out_dir / model_file
    kwargs = dict(
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    was_training = model.training
    try:
        # Pause the memory: its hooks must not fire (and mutate statistics)
        # while the tracer runs the dummy batch.
        with memory.paused():
            try:
                torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, **kwargs)
            except TypeError:  # older torch without the dynamo flag
                torch.onnx.export(wrapper, dummy, str(onnx_path), **kwargs)
    finally:
        wrapper.remove_hooks()
        model.train(was_training)

    mean, std = spec.normalization()
    manifest = {
        "format": "hatchvision-bundle",
        "version": "1.0",
        "model": model_file,
        "graph": graph_file,
        "dataset": spec.name,
        "image_size": spec.image_size,
        "in_channels": spec.in_channels,
        "mean": list(mean),
        "std": list(std),
        "class_names": list(spec.class_names),
        "activation_outputs": [
            {
                "layer": name,
                "output": f"act_{name}",
                "units": memory.stats[name].dim,
                "node_prefix": f"u:{name}:",
            }
            for name in tracked
        ],
    }
    manifest.update(extra_meta or {})
    manifest_path = out_dir / manifest_file
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
