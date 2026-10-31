"""A checkpoint folder: ``config.json``, safetensors (one file or an index of shards) read lazily,
and the quantization scheme of each linear (``quantization_config``: compressed-tensors or
NVIDIA ModelOpt).

``QuantScheme.part(ckpt, name, device)`` reads one linear in its format with the scales in one
convention: ``weight`` (bf16 ``[N, K]``; ``mxfp8`` / ``fp8`` / ``fp8_dynamic`` E4M3 ``[N, K]``;
``nvfp4`` packed uint8 ``[N, K / 2]``), ``block`` (row-major block scales: ue8m0 per 32 for
``mxfp8``, E4M3 per 16 for ``nvfp4``, float32 per 128 for ``fp8_dynamic``, any row its own),
``wmul`` (the weight's tensor scale, ``w ~ codes * block * wmul``) and
``imul`` (the static activation scale, ``x ~ codes * block * imul``), each a 1-element float32.
"""

import fnmatch
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from tokenomos.kernels.linear import fp8

# compressed-tensors format -> our weight format
FORMATS = {
    "mxfp8-quantized": "mxfp8",
    "nvfp4-pack-quantized": "nvfp4",
    "float-quantized": "fp8_dynamic",
}
# ModelOpt algorithm -> our weight format
MODELOPT = {"NVFP4": "nvfp4", "FP8": "fp8"}
# compressed-tensors targets naming a module class rather than a module path
CLASS_TARGETS = {"Linear"}


class QuantScheme:
    """Linear name (``...layers.3.conv.in_proj``) -> weight format (``bf16`` when nothing
    targets it or it is ignored). compressed-tensors targets: exact names, ``re:`` patterns, or a
    class name (``Linear``: every linear); ModelOpt: one algorithm for every linear but the
    ``ignore`` globs, or per layer (``MIXED_PRECISION``'s ``quantized_layers``)."""

    def __init__(self, config: dict | None):
        self.method = None
        self.groups: list[tuple[list[str], str]] = []
        self.ignore: list[str] = []
        self.layers: dict[str, str] = {}
        self.default = "bf16"
        if not config:
            return
        self.method = config.get("quant_method")
        if self.method == "compressed-tensors":
            for name, group in config.get("config_groups", {}).items():
                fmt = group.get("format") or config.get("format")
                if fmt not in FORMATS:
                    raise ValueError(f"quantization group {name}: format {fmt!r} is not supported")
                self.groups.append((group["targets"], FORMATS[fmt]))
            self.ignore = list(config.get("ignore", []))
        elif self.method == "modelopt":
            algo = config.get("quant_algo")
            self.ignore = list(config.get("ignore", []))
            if algo == "MIXED_PRECISION":
                for name, spec in config["quantized_layers"].items():
                    if spec["quant_algo"] not in MODELOPT:
                        raise ValueError(f"{name}: ModelOpt {spec['quant_algo']!r} not supported")
                    self.layers[name] = MODELOPT[spec["quant_algo"]]
            elif algo in MODELOPT:
                self.default = MODELOPT[algo]
            else:
                raise ValueError(f"ModelOpt quant_algo {algo!r} is not supported")
        else:
            raise ValueError(f"quantization {self.method!r} is not supported")

    @staticmethod
    def _matches(pattern: str, name: str) -> bool:
        if pattern.startswith("re:"):
            return re.match(pattern[3:], name) is not None
        return pattern in CLASS_TARGETS or pattern == name

    def format_of(self, name: str) -> str:
        if self.method == "modelopt":
            if self.layers:
                return self.layers.get(name, "bf16")
            if any(fnmatch.fnmatchcase(name, p) for p in self.ignore):
                return "bf16"
            return self.default
        if any(self._matches(p, name) for p in self.ignore if p not in CLASS_TARGETS):
            return "bf16"
        for targets, fmt in self.groups:
            if any(self._matches(p, name) for p in targets):
                return fmt
        return "bf16"

    def part(self, ckpt: "Checkpoint", name: str, device) -> dict:
        """Linear ``name`` (no ``.weight``) in its format (see the module docstring); a
        targeted linear stored without scales is bf16 (LFM3-VL's vision layers match the text
        layers' patterns but are bf16)."""
        fmt = self.format_of(name)
        if not ckpt.has(f"{name}.weight_scale"):
            fmt = "bf16"

        def get(suffix: str) -> torch.Tensor:
            return ckpt.get(f"{name}.{suffix}", device)

        def scalar(t: torch.Tensor) -> torch.Tensor:
            return t.float().reshape(1)

        if fmt == "nvfp4" and self.method == "compressed-tensors":
            return {
                "fmt": fmt,
                "weight": get("weight_packed"),
                "block": get("weight_scale"),
                "wmul": 1.0 / scalar(get("weight_global_scale")),
                "imul": 1.0 / scalar(get("input_global_scale")),
            }
        if fmt == "nvfp4":
            return {
                "fmt": fmt,
                "weight": get("weight"),
                "block": get("weight_scale"),
                "wmul": scalar(get("weight_scale_2")),
                "imul": scalar(get("input_scale")),
            }
        if fmt == "fp8":
            return {
                "fmt": fmt,
                "weight": get("weight"),
                "wmul": scalar(get("weight_scale")),
                "imul": scalar(get("input_scale")),
            }
        if fmt == "fp8_dynamic":
            weight = get("weight")
            block = fp8.expand_scales(get("weight_scale"), *weight.shape)
            return {"fmt": fmt, "weight": weight, "block": block}
        if fmt == "mxfp8":
            return {"fmt": fmt, "weight": get("weight"), "block": get("weight_scale")}
        return {"fmt": "bf16", "weight": get("weight")}


class Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text("utf-8"))
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text("utf-8"))["weight_map"]
        else:
            with safe_open(self.path / "model.safetensors", "pt") as f:
                weight_map = dict.fromkeys(f.keys(), "model.safetensors")
        self.weight_map: dict[str, str] = weight_map
        self._files: dict[str, object] = {}
        self.quant = QuantScheme(self.config.get("quantization_config"))

    @property
    def architecture(self) -> str:
        return self.config["architectures"][0]

    @property
    def text_config(self) -> dict:
        return self.config.get("text_config", self.config)

    @property
    def tied(self) -> bool:
        """Whether the head is the embedding: the top-level flag wins over the text config's."""
        if "tie_word_embeddings" in self.config:
            return self.config["tie_word_embeddings"]
        return self.text_config.get("tie_word_embeddings", True)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str, device: torch.device | str = "cpu") -> torch.Tensor:
        file = self.weight_map[name]
        if file not in self._files:
            self._files[file] = safe_open(self.path / file, "pt", device="cpu")
        return self._files[file].get_tensor(name).to(device)
