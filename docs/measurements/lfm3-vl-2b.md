# LFM3-VL 2B (MXFP8): measured

RTX 5090, checkpoint `lfm3-vl-2b-mxfp8` with the MXFP8 LM head.
Baseline: vLLM 0.32.0 + KohakuvLLM (torch 2.13.0+cu130, flashinfer 0.7.0.post1, triton 3.7.1),
`max_num_seqs` 256, `max_num_batched_tokens` 8192, prefix caching off. Every number below comes
from a script in `bench/` whose report lands in `reports/`.

## Stock vLLM

vLLM has no LFM3. `bench/suite/run.py --box 5090 --only 5090_wall,5090_raw` runs vLLM 0.32.0 with
LFM3's model definition registered as an architecture and nothing else of the plugin: every linear,
attention and sampler kernel is vLLM's own, and the width-7 conv is the plugin's, since vLLM's
`causal_conv1d` kernels stop at width 5. S x 64-token prompts, 512 generated; S x 2048-token
prompts, 1 generated. `async_scheduling=True` changes nothing: 16,734 tok/s at 512.

Multiprocess engine, FP8 checkpoint, tok/s over the whole call, with nsys step ms wall / busy / idle
from an in-process engine (`bench/suite/vllm_raw.py`, `bench/profile/nsys_steps.py`):

| seqs | prefill tok/s | decode tok/s | decode tok/s, in process | step ms |
|---|---|---|---|---|
| 8 | 111,850 | 3,194 | 2,176 | 3.58 / 2.31 / 1.27 |
| 32 | 119,640 | 8,356 | 6,045 | 5.10 / 2.60 / 2.50 |
| 64 | 117,967 | 11,198 | 9,004 | 6.96 / 2.93 / 4.03 |
| 128 | 118,148 | 13,577 | 11,468 | 10.75 / 3.83 / 6.92 |
| 256 | 124,554 | 15,813 | 13,178 | 18.50 / 6.12 / 12.38 |
| 512 | 120,229 | 16,557 | 14,665 | 33.98 / 10.77 / 23.21 |

The in-process engine on other setups, tok/s at 512 and step ms: MXFP8 checkpoint 14,881, 33.33 /
10.00 / 23.33; with `KOHAKU_MXFP8=1` 14,906, 33.33 / 9.93 / 23.40. Prefill at 512: 108,797 and
121,040.

## Text: decode and prefill against KohakuvLLM (1 GPU)

Random prompts, greedy, EOS ignored, 8192 tokens per prefill forward. Decode: S × 64-token
prompt, 512 generated. Prefill: S × 2048-token prompts, 1 generated. Step ms: wall / GPU busy /
GPU idle per decode step (nsys). Host CPU: Tokenomos's host time per step. (`bench/decode.py`,
`bench/profile/nsys_steps.py`)

The vLLM columns here come from an earlier KohakuvLLM configuration that was not recorded. The
current `KOHAKU_MXFP8=1` arm above does not reproduce them: 14,906 tok/s at 512 against 22,448.

| seqs | prefill tok/s vLLM | prefill tok/s Tokenomos | decode tok/s vLLM | decode tok/s Tokenomos | step ms vLLM | step ms Tokenomos | host ms Tokenomos |
|---|---|---|---|---|---|---|---|
| 8 | 97,594 | 127,946 | 2,773 | 4,129 | 2.88 / 2.60 / 0.28 | 1.94 / 1.85 / 0.09 | 0.12 |
| 32 | 123,127 | 155,755 | 7,863 | 15,648 | 4.07 / 2.68 / 1.39 | 2.04 / 1.96 / 0.08 | 0.12 |
| 64 | 124,532 | 155,772 | 10,552 | 29,257 | 6.07 / 2.70 / 3.37 | 2.19 / 2.10 / 0.09 | 0.13 |
| 128 | 131,996 | 155,607 | 17,261 | 48,726 | 7.42 / 3.24 / 4.18 | 2.63 / 2.49 / 0.13 | 0.18 |
| 256 | 136,727 | 155,660 | 20,416 | 66,298 | 12.54 / 5.00 / 7.54 | 3.86 / 3.71 / 0.16 | 0.26 |
| 512 | 137,803 | 155,061 | 22,448 | 79,030 | 22.81 / 8.45 / 14.37 | 6.48 / 6.25 / 0.23 | 0.41 |

Decode at 256 sequences against the hardware (`bench/profile/roofline.py`): GEMMs 2.03 ms
against a 1.24 ms compute floor (narrow-N GEMMs at 25-33 % of peak); attention at 89-98 % of
copy bandwidth (1758 GB/s).

## Root captions (vision + prefill + decode)

The first N rows of `calib_rows.jsonl` (GBC root-caption chats: system
prompt + one image as a data URL), greedy, thinking off, images capped at 1024² px, 256 slots /
`max_num_seqs` per GPU. Mean prompt 1083.6 tokens, mean output ~100 tokens, every output ends on
a stop token. Wall time covers image decode and resize, vision, prefill and decode.
(`bench/caption.py`, `bench/serve_caption.py`, `lfm3_test/vl_caption.py`)

| GPUs | rows | path | vLLM img/s | Tokenomos img/s | ratio |
|---|---|---|---|---|---|
| 1 | 256 | in-Python `LLM.chat` | 36.75 | 69.0 | 1.88 |
| 1 | 256 | OpenAI HTTP, 256 concurrent | 40.47 | 69.1 | 1.71 |
| 4 | 256 | OpenAI HTTP (vLLM DP 4 / router) | 115.2 | 158.4 | 1.38 |
| 4 | 1024 | OpenAI HTTP | 187.3 | 291.9 | 1.56 |
| 4 | 1345 | OpenAI HTTP | 191.7 | 262.2 | 1.37 |

Rows beyond the slot count (1345 over 4 × 256) wait for a second wave whose tail runs at small
batches: on 1 GPU, 336 rows give 68.7 img/s with 256 slots and 79.1 img/s with 384 slots.

The rows above ran without the prefix table. With it (64 entries; every row shares the system
prompt, the share ends at the first image), 1 GPU, in-Python, 256 rows, back-to-back runs:

| prefix table | img/s | prefill tokens | prefill GPU s | hits |
|---|---|---|---|---|
| off | 72.2 | 277,398 | 1.72 | - |
| on | 103.7 | 90,550 | 0.70 | 249 |

Replies with the table are as faithful as without (64 teacher-forced, MXFP8 reference: argmax
agreement 0.9854 vs 0.9832, gap max 1.125 vs 1.875). Under eviction pressure (bf16, 8 distinct
starts, 3 entries, 48 replies) the device free stack equals the host's page count after every
round, and hit replies score argmax 0.9948, gap max 0.188 against the dense reference.

Where one GPU's caption time goes (256 rows, nsys over the timed call, `bench/profile/
nsys_timeline.py`): GPU busy 96 % of 3.59 s; prefill 1.71 s, decode 1.23 s, vision 0.52 s.

Host preparation per conversation, one thread (`bench/prep.py`): template 0.26 ms, tokenize
1.9 ms, image decode 1.5 ms, resize 1.3 ms. The prep pool (whole conversations in workers,
images through the shared pinned ring) runs 1247 rows/s with 16 workers.

## Output faithfulness

64 captions teacher-forced through the dense single-sequence reference (same MXFP8 weights,
`bench/caption_gaps.py`): gap = max log p − log p(chosen) per generated token.

| run | tokens | argmax agreement | gap mean | gap max | gaps > 0.15 | gaps > 1 |
|---|---|---|---|---|---|---|
| Tokenomos run A | 8925 | 0.9844 | 0.0051 | 1.875 | 93 | 3 |
| Tokenomos run B | 8581 | 0.9824 | 0.0049 | 1.125 | 91 | 1 |
| vLLM | 8709 | 0.9835 | 0.0055 | 2.5 | 84 | 5 |

Both engines sit at the same distance from the reference; outputs of two runs differ (batch
composition changes the MXFP8 GEMM shapes) at tokens whose two candidates are within 1.25 nats.
