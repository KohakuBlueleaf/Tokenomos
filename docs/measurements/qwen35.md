# Qwen3.5 / Qwen3.8: measured

RTX 5090, one GPU per run. Baseline: vLLM 0.32.0 + KohakuvLLM
(torch 2.13.0+cu130, flashinfer 0.7.0.post1, triton 3.7.1). Checkpoints: `qwen35-4b-bf16`,
`qwen35-4b-nvfp4` / `qwen35-9b-nvfp4` (ModelOpt NVFP4, every linear but the head),
`qwen38-27b-nvfp4-full` (compressed-tensors NVFP4; GDN `in_proj_a` / `in_proj_b` and the head bf16).

## Text: decode and prefill (1 GPU)

Random prompts, greedy, EOS ignored. Decode: S × 64-token prompts, 512 generated, tok/s = every
generated token over the call's wall time (both engines). Prefill: S × 2048-token prompts, 1
generated. vLLM: `gpu_memory_utilization` 0.85 (27B 0.90), prefix caching off. Tokenomos: tuned
NVFP4 GEMM table (`bench/profile/tune_nvfp4.py`), FlashInfer GDN prefill, bf16 head (27B NVFP4).
(`bench/decode.py`, `lfm3_test/lm_bench.py`)

| model | state | seqs | decode vLLM | decode Tokenomos | prefill vLLM | prefill Tokenomos |
|---|---|---|---|---|---|---|
| 4B | int8 | 8 | 2,183 | 2,594 | 63,385 | 98,549 |
| 4B | int8 | 64 | 12,775 | 14,875 | 86,085 | 102,003 |
| 4B | int8 | 256 | 19,506 | 24,411 | 89,331 | 102,043 |
| 4B | int8 | 512 | 21,146 | 25,884 | 89,442 | 101,686 |
| 4B | fp32 | 8 | 2,063 | 2,405 | 64,904 | 97,624 |
| 4B | fp32 | 256 | 11,677 | 11,962 | 89,942 | 99,154 |
| 9B | int8 | 8 | 1,483 | 1,699 | 47,837 | 61,753 |
| 9B | int8 | 64 | 9,807 | 10,321 | 56,252 | 62,396 |
| 9B | int8 | 256 | 17,714 | 18,825 | 57,390 | 62,136 |
| 9B | int8 | 512 | 19,034 | 16,673 * | 57,104 | 61,869 |
| 27B | int8 | 8 | 694 | 735 | 17,231 | 18,732 |
| 27B | int8 | 64 | 4,223 | 4,337 | 18,466 | 18,871 |
| 27B | int8 | 128 | 3,876 | 4,178 * | 18,579 | 18,816 |

\* page reservation (prompt + `max_new` per row) holds 389 (9B) / 75 (27B) rows at once; vLLM
admits every row and preempts. Where the step goes (27B prefill, 32 × 2048, nsys): FP4 GEMMs
2.27 s on both engines, total GPU 3.34 s ours vs 3.35 s vLLM's.

## Parity with Hugging Face (bf16 4B, the box's `transformers`, bf16)

Text, random token prompts of 7 / 64 / 65 / 200 / 513 tokens packed in one reference forward:
argmax agreement 0.97-1.00, gap (HF max log p - HF log p of our argmax) at most 0.125.

Images (GBC caption rows, max 1024² px): prepared size, pixels, prompt ids and the multimodal
RoPE positions equal Hugging Face's exactly. The vision tower amplifies bf16 rounding: each block
alone matches Hugging Face's within 0.27 % (relative L2), the 24-block output is 7.75 % from
Hugging Face's fp32 tower, Hugging Face's own bf16 tower 7.58 %.

## GDN state formats (engine, bf16 4B)

16 chat prompts sharing a 140-token system prompt (prefix table on: 8 hits), greedy, up to 192
tokens; each reply teacher-forced through the dense reference with the fp32 state.

| state | GDN state per slot (4B) | argmax agreement | gap mean | gap max |
|---|---|---|---|---|
| fp32 | 48 MiB | 0.9950 | 0.0009 | 0.250 |
| bf16 | 24 MiB | 0.9994 | 0.0001 | 0.125 |
| int8 | 12.4 MiB | 0.9918 | 0.0015 | 0.375 |
| fp8 | 12.4 MiB | 0.9851 | 0.0027 | 0.375 |

## Root captions

The first 256 rows of `calib_rows.jsonl` (system prompt + one image),
greedy, thinking off, images capped at 1024² px, `max_tokens` 1024; mean prompt 1084.4 tokens,
mean reply 120-145 tokens. In-Python (`bench/caption.py`, `lfm3_test/vl_caption.py`). Tokenomos:
`kv_fraction` 0.85, prefix table 8 entries; vLLM: `max_num_seqs` 256, `max_num_batched_tokens`
8192, `gpu_memory_utilization` 0.85, prefix caching off (27B: the GBC service's settings below).

| model | engine | GDN state | head | slots / seqs | img/s |
|---|---|---|---|---|---|
| 4B NVFP4 | vLLM | fp32 | bf16 | 256 | 26.77 |
| 4B NVFP4 | vLLM (`KOHAKU_GDN_STATE=int8`) | int8 | bf16 | 256 | 36.08 |
| 4B NVFP4 | Tokenomos | int8 | bf16 | 256 | 43.88 / 48.81 / 41.41 |
| 4B NVFP4 | Tokenomos | int8 | MXFP8 | 256 | 44.30 / 49.61 / 33.02 * |
| 4B NVFP4 | Tokenomos | bf16 | bf16 | 256 | 28.08 * / 30.42 * |
| 9B NVFP4 | vLLM | fp32 | bf16 | 256 | 19.91 |
| 9B NVFP4 | vLLM (`KOHAKU_GDN_STATE=int8`) | int8 | bf16 | 256 | 26.40 |
| 9B NVFP4 | Tokenomos | int8 | bf16 | 256 | 35.15 |
| 27B NVFP4 full | vLLM (`KOHAKU_NVFP4=norm,quant,head=nvfp4`, int8 state) | int8 | NVFP4 | 64 | 4.71 |
| 27B NVFP4 full | Tokenomos | int8 | NVFP4 | 64 | 8.08 |

\* runs with a reply that repeats until `max_tokens` (one row runs about 600 decode steps alone
at the tail): 0-2 such rows per run of ours, none in vLLM's four runs. Both engines branch at the
same early tokens (see faithfulness); the rate needs more rows to compare. 27B settings (the GBC
service's): vLLM `gpu_memory_utilization` 0.90, `kv_cache_memory_bytes` 4 GiB, `max_num_seqs` 64,
`max_num_batched_tokens` 4096; Tokenomos 64 slots, `prefill_budget` 4096, 4 prefix entries.

Where one 4B run's time goes (int8 state, 43.88 img/s, 5.83 s): 16 prefill forwards (about 90k
tokens after the prefix table, 1.32 s GPU), vision 1.10 s GPU, 528 decode steps. The 27B run took 151 prefill
forwards: admission reserves pages for prompt + `max_tokens`, so at 64 slots the reservation, not
the GPU, limits how many prompts one forward takes.

## Faithfulness (4B NVFP4)

32 captions of each engine teacher-forced through our dense reference with the same NVFP4
weights (fp32 state, `bench/caption_gaps.py`):

| run | tokens | argmax agreement | gap mean | gap max | gaps > 0.15 | gaps > 1 |
|---|---|---|---|---|---|---|
| Tokenomos (fp32 state) | 7029 | 0.9384 | 0.0477 | 11.25 | 362 | 95 |
| vLLM | 6785 | 0.9378 | 0.0439 | 6.25 | 352 | 78 |

NVFP4 activations move the argmax at 6 % of tokens for both engines alike (their NVFP4 kernels
round differently from the reference's); the largest gaps sit at the reply's first tokens, where
it picks between a fenced and a bare JSON object.

## Kernels

NVFP4 linears against a float64 decode of the stored codes and scales (low nibble the even
column), the activation quantized and decoded with the checkpoint's static scale: relative error
0.0017 (the bf16 output rounding) for every fused group of every checkpoint (NVFP4, FP8, stacked
FP8 + bf16, the NVFP4 head). The GDN chunked prefill (FlashInfer fused, the default; vendored
FLA) against a float32 token recurrence, 8 × 1024 tokens with initial states: output error 0.0037
for both, final state 0.0033 / 0.0030; FlashInfer 271 us vs FLA 781 us (4B heads), 402 vs 1131
(27B). The tuned NVFP4 table picks the fastest CUTLASS tactic or b12x plan per shape and token
count, timed over weight copies past the L2: 2.1-2.6x the default tactic at 2048+ tokens, equal
at 8. The FP4 GEMM is row-independent for M 1 .. 1024 on the CUTLASS, b12x
and cuDNN backends. The fused producers (norm + quantize, `silu * up`, the output gates, the GDN
gated norm) write the same bytes as their unfused torch compositions quantized after. FlashInfer's
`mm_fp4` front end costs 346 us of host time a call; the CUTLASS op called directly 28 us.
