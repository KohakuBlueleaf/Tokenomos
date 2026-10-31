"""vLLM's decode or prefill on random tokens. With the engine core in this process
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``) ``cudaProfilerStart`` / ``Stop`` bracket the timed call
under ``nsys profile --capture-range=cudaProfilerApi``. Decode: S x 64-token prompts, 512 generated.
Prefill: S x 2048-token prompts, 1 generated. Merges the mode's numbers into ``--out``.

    python bench/suite/vllm_raw.py --model M --seqs 256 --mode decode --out reports/vllm_raw_x.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqs", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["decode", "prefill"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-kw", default="{}", help="JSON kwargs of LLM")
    args = ap.parse_args()
    length, steps = (64, 512) if args.mode == "decode" else (2048, 1)
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        max_num_seqs=args.seqs,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        mamba_ssm_cache_dtype="bfloat16",
        trust_remote_code=True,
        **json.loads(args.engine_kw),
    )
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)
    prompts = [
        TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab - 1000) for _ in range(length)])
        for _ in range(args.seqs)
    ]
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    t0 = time.perf_counter()
    llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    torch.cuda.cudart().cudaProfilerStop()
    out = Path(args.out)
    report = json.loads(out.read_text("utf-8")) if out.exists() else {"seqs": args.seqs}
    report[f"{args.mode}_wall_s"] = round(wall, 3)
    if args.mode == "decode":
        report["decode_tok_s_wall"] = round(args.seqs * steps / wall, 1)
    else:
        report["prefill_tok_s"] = round(args.seqs * length / wall, 1)
    out.write_text(json.dumps(report, indent=1), "utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
