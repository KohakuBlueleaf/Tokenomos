"""Every arm of a full measurement, per box. A card class is a set of cards that take the same arms;
the tests and the GDN tile sweep open a class, everything else runs after them.

``5090``: the 3 x 2 caption table (Tokenomos, KohakuvLLM, stock vLLM; in process and HTTP, 4096
``max_len``, 1024 in flight), FlashInfer's GDN prefill against ``fused``. ``p8``: decode, prefill
and captions per work dtype and W8A8 on the RTX 3090 and 2080 Ti, one full GBC graph. ``win``: the
same on the RTX 4090, FP8 included, under a 19 GiB cap.
"""

from dataclasses import dataclass, field


@dataclass
class Arm:
    tag: str
    card: str
    # tests, tune, decode, caption, serve, vllm, vllm_serve, gbc
    kind: str
    model: str = ""
    arch: dict = field(default_factory=dict)
    engine: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    # decode: sequence count, prefill sequence count
    seqs: int = 0
    prefill_seqs: int = 0
    kohaku: bool = False
    limit: int = 1800


CAPTION = {"max_len": 4096}
HEAD_MX, HEAD_NV, HEAD_BF = {"head": "mxfp8"}, {"head": "nvfp4"}, {"head": "bf16"}
INT8 = {"state_format": "int8"}
TOKENOMOS_5090 = {
    "lfm3": ("lfm3-vl-2b-mxfp8", HEAD_MX),
    "q4": ("qwen35-4b-mlp4-mxfp8", {**HEAD_MX, **INT8}),
    "e4b": ("gemma4-e4b-it-nvfp4", HEAD_NV),
    "q9": ("qwen35-9b-nvfp4", {**HEAD_BF, **INT8}),
    "g12": ("gemma4-12b-it-nvfp4", HEAD_NV),
    "q27": ("qwen38-27b-nvfp4-full", {**HEAD_NV, **INT8}),
    "g31": ("gemma4-31b-it-nvfp4-redhat", HEAD_NV),
}
# a slot's ring binds these two: what the automatic reserve keeps past their prefill goes to slots
RESERVE = {"g12": {"kv_reserve": 4 << 30}, "g31": {"kv_reserve": 2 << 30}}
GDN_STATE = {"KOHAKU_GDN_STATE": "int8"}
KOHAKU_5090 = {
    "lfm3": ("lfm3-vl-2b-mxfp8", {}),
    "q4": ("qwen35-4b-mlp4-mxfp8", GDN_STATE),
    "q9": ("qwen35-9b-nvfp4", GDN_STATE),
    "q27": ("qwen38-27b-nvfp4-full", {**GDN_STATE, "KOHAKU_NVFP4": "norm,quant,head=nvfp4"}),
}
STOCK_5090 = {
    "q4": "qwen35-4b-mlp4-fp8",
    "e4b": "gemma4-e4b-it-nvfp4",
    "q9": "qwen35-9b-nvfp4",
    "g12": "gemma4-12b-it-nvfp4",
    "q27": "qwen38-27b-nvfp4-full",
    "g31": "gemma4-31b-it-nvfp4-redhat",
}
# tag -> (model, arch, the card classes that hold its dtypes)
ROTATE = {"int8": True, "rotate": 16}
BOTH, AMPERE = ("3090", "2080"), ("3090",)
P8 = {
    "lfm_bf16": ("lfm3-vl-2b", {"work": "bf16"}, AMPERE),
    "lfm_fp16": ("lfm3-vl-2b", {"work": "fp16"}, BOTH),
    "lfm_fp32_cublas": ("lfm3-vl-2b", {"work": "fp32", "route": "cublas"}, BOTH),
    "lfm_fp32_tilelang": ("lfm3-vl-2b", {"work": "fp32", "route": "tilelang"}, BOTH),
    "lfm_w8a8": ("lfm3-vl-2b", ROTATE, BOTH),
    "q4_bf16": ("qwen35-4b-bf16", {"work": "bf16"}, AMPERE),
    "q4_fp16": ("qwen35-4b-bf16", {"work": "fp16"}, BOTH),
    "q4_fp16_fla": ("qwen35-4b-bf16", {"work": "fp16", "gdn_prefill": "fla"}, BOTH),
    "q4_fp32_cublas": ("qwen35-4b-bf16", {"work": "fp32", "route": "cublas"}, BOTH),
    "q4_fp32_tilelang": ("qwen35-4b-bf16", {"work": "fp32", "route": "tilelang"}, BOTH),
    "q4_w8a8": ("qwen35-4b-bf16", {**ROTATE, **INT8}, BOTH),
    "e4b_bf16": ("gemma4-e4b-it", {"work": "bf16"}, AMPERE),
    "e4b_fp16": ("gemma4-e4b-it", {"work": "fp16"}, BOTH),
    "e4b_w8a8": ("gemma4-e4b-it", ROTATE, BOTH),
}
LEAVE = {"kv_leave": 3 << 29, "kv_reserve": 5 << 30}
Q4 = "Qwen--Qwen3.5-4B"
Q9 = "RedHatAI--Qwen3.5-9B-FP8-dynamic"
WIN = {
    "lfm_bf16": ("lfm3-vl-2b", HEAD_BF),
    "lfm_w8a8": ("lfm3-vl-2b", {**HEAD_BF, **ROTATE}),
    "lfm_fp8": ("lfm3-vl-2b-fp8", HEAD_BF),
    "q4_bf16": (Q4, {**HEAD_BF, **INT8}),
    "q4_bf16_fla": (Q4, {**HEAD_BF, **INT8, "gdn_prefill": "fla"}),
    "q4_w8a8": (Q4, {**HEAD_BF, **INT8, **ROTATE}),
    "q4_fp8": ("qwen35-4b-fp8", {**HEAD_BF, **INT8}),
    "q4_fp8_gdn": ("qwen35-4b-fp8-gdn", {**HEAD_BF, **INT8}),
    "q4_fp8_state": ("qwen35-4b-fp8-gdn", {**HEAD_BF, "state_format": "fp8"}),
    "q9_fp8": (Q9, {**HEAD_BF, **INT8}),
    "q9_fp8_state": (Q9, {**HEAD_BF, "state_format": "fp8"}),
    "e4b_bf16": ("gemma4-e4b-it", HEAD_BF),
    "e4b_w8a8": ("gemma4-e4b-it", {**HEAD_BF, **ROTATE}),
}
WIN_HTTP = ("lfm_bf16", "lfm_w8a8", "q4_w8a8", "q9_fp8")


def box_5090() -> list[Arm]:
    arms = [Arm("5090_tests", "5090", "tests"), Arm("5090_tune", "5090", "tune")]
    for name, (model, arch) in TOKENOMOS_5090.items():
        engine = {**CAPTION, **RESERVE.get(name, {})}
        arms.append(Arm(f"5090_t_{name}", "5090", "caption", model, arch, engine))
        arms.append(Arm(f"5090_t_http_{name}", "5090", "serve", model, arch, engine))
    for name, (model, env) in KOHAKU_5090.items():
        arms.append(Arm(f"5090_k_{name}", "5090", "vllm", model, env=env, kohaku=True))
        arms.append(Arm(f"5090_k_http_{name}", "5090", "vllm_serve", model, env=env, kohaku=True))
    for name, model in STOCK_5090.items():
        arms.append(Arm(f"5090_s_{name}", "5090", "vllm", model))
        arms.append(Arm(f"5090_s_http_{name}", "5090", "vllm_serve", model))
    for prefill in ("flashinfer", "fused"):
        for name in ("q4", "q27"):
            model, arch = TOKENOMOS_5090[name]
            arch = {**arch, "gdn_prefill": prefill}
            tag = f"5090_gdn_{name}_{prefill}"
            arms.append(Arm(tag, "5090", "decode", model, arch, seqs=64, prefill_seqs=64))
    return arms


def box_p8() -> list[Arm]:
    arms = []
    for card in BOTH:
        arms += [Arm(f"{card}_tests", card, "tests"), Arm(f"{card}_tune", card, "tune")]
    for name, (model, arch, cards) in P8.items():
        for card in cards:
            seqs = 128 if card == "3090" else 64
            tag = f"{card}_{name}"
            arms.append(Arm(tag, card, "decode", model, arch, seqs=seqs, prefill_seqs=seqs))
            limit = 3600 if card == "2080" else 1800
            arms.append(Arm(f"{tag}_cap", card, "caption", model, arch, CAPTION, limit=limit))
    q4 = '{"int8": true, "rotate": 16, "state_format": "int8"}'
    arms.append(Arm("3090_gbc_q4_w8a8", "3090", "gbc", "qwen35-4b-bf16", {"json": q4}, limit=1700))
    return arms


def box_win() -> list[Arm]:
    arms = [Arm("4090_tests", "4090", "tests", limit=3600), Arm("4090_tune", "4090", "tune")]
    for name, (model, arch) in WIN.items():
        for seqs in (32, 64, 128, 256) if name.startswith("e4b") else (32, 128, 256):
            tag = f"4090_{name}_n{seqs}"
            arms.append(Arm(tag, "4090", "decode", model, arch, LEAVE, seqs=seqs, prefill_seqs=32))
        engine = {**CAPTION, **LEAVE}
        arms.append(Arm(f"4090_{name}_cap", "4090", "caption", model, arch, engine))
        if name in WIN_HTTP:
            arms.append(Arm(f"4090_{name}_http", "4090", "serve", model, arch, engine))
    return arms


BOXES = {"5090": box_5090, "p8": box_p8, "win": box_win}
