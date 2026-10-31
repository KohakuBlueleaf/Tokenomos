# Root captions

1024 GBC root-caption rows (`calib_rows.jsonl`), greedy, thinking off, `max_tokens` 1024, one
RTX 5090 32 GB per run. Images per second over the whole batch. Long side of the rows' images: 74
to 1568 px, median 391, median 0.08 Mpx.

Every engine and call path runs the same rows on the same box, `max_len` 4096, 1024 requests in
flight; `bench/suite/run.py --box 5090` runs them all and writes `reports/suite/5090/summary.jsonl`.

- Tokenomos in process: `bench/caption.py`, automatic sizing, `LLM.chat`, 16 prep workers.
- Tokenomos HTTP: `tokenomos.api.server`, `bench/serve_caption.py`.
- vLLM in process: `bench/suite/vllm_caption.py`, `LLM.chat`, the checkpoint's generation config.
- vLLM HTTP: `vllm serve`, `bench/serve_caption.py`.
- vLLM on both paths: 0.90 of the card, 256 sequences, prefix caching on. Stock runs with no plugin
  and the bf16 GDN state; KohakuvLLM with the int8 GDN state, 27B also NVFP4 norm, quant and head.
  Stock 27B in process holds 192 sequences: 256 exceed its Mamba cache blocks.

| Model | Tokenomos in process | Tokenomos HTTP | KohakuvLLM in process | KohakuvLLM HTTP | Stock vLLM in process | Stock vLLM HTTP |
|---|---|---|---|---|---|---|
| LFM3-VL 2B MXFP8 | 130.30 | 135.05 | 8.08 (loops) | 34.37 | 46.89 (FP8) | 42.44 (FP8) |
| Qwen3.5 4B MXFP8 | 64.91 | 62.17 | 22.34 | 20.20 | 23.96 (FP8) | 21.79 (FP8) |
| Gemma 4 E4B NVFP4 | 58.93 | 55.57 | not run | not run | 21.30 | 20.04 |
| Qwen3.5 9B NVFP4 | 48.99 | 46.43 | 21.80 | 20.82 | 21.87 | 18.09 |
| Gemma 4 12B NVFP4 | 22.88 | 22.70 | not run | not run | 21.06 | 20.71 |
| Qwen3.8 27B NVFP4 | 18.40 | 18.85 | 4.94 | 4.88 | 3.85 | 4.26 |
| Gemma 4 31B NVFP4 | 3.60 | 3.61 | not run | not run | 3.25 | 3.79 |

12B and 31B set `kv_reserve` to 4 and 2 GiB: the automatic reserve leaves them 41 and 7 slots, 22.44
and 3.22 img/s. Tokenomos's LFM3 in-process row holds one runaway reply.

vLLM has no LFM3: its rows register LFM3's model definition, with vLLM's own linear, attention and
sampler kernels and the plugin's width-7 conv. In process and over HTTP per checkpoint: FP8 46.89 /
42.44, NVFP4 36.98 / 7.88, MXFP8 27.75 / 32.07, bf16 33.86 / 8.10. NVFP4 and bf16 over HTTP loop on
982 and 1005 replies. KohakuvLLM (`KOHAKU_MXFP8=1`) in process loops on 1014 replies; its HTTP row
stops on all 1024. Both stock LFM3 and KohakuvLLM apply the generation config's repetition penalty
as `vllm serve` does.

## Where the time goes (Tokenomos in process)

| Model | Slots | Pages | Wall s | Encode GPU s | Prefill GPU s | Decode wait s | Live rows | Prefill forwards | Peak reserved GiB |
|---|---|---|---|---|---|---|---|---|---|
| LFM3-VL 2B | 256 | 32769 | 7.9 | 2.1 | 2.3 | 4.5 | 71.3 | 50 | 18.18 |
| Qwen3.5 4B | 256 | 16625 | 15.8 | 4.8 | 4.8 | 9.3 | 179.2 | 50 | 29.26 |
| Gemma 4 E4B | 256 | 19967 | 17.4 | 7.8 | 4.7 | 9.2 | 177.5 | 54 | 30.58 |
| Qwen3.5 9B | 256 | 12357 | 20.9 | 7.7 | 5.7 | 10.1 | 169.4 | 72 | 28.46 |
| Gemma 4 12B | 43 | 3027 | 44.8 | 0.2 | 11.6 | 31.7 | 40.0 | 199 | 28.92 |
| Qwen3.8 27B | 67 | 2709 | 55.6 | 8.0 | 18.5 | 33.6 | 60.7 | 161 | 27.93 |
| Gemma 4 31B | 9 | 511 | 284.8 | 27.8 | 75.0 | 193.1 | 8.9 | 1001 | 30.43 |

Live rows: generated tokens over decode steps. A step's cost follows the bucket, so rows below the
slot count are paid for and idle. The pause path did not fire on any row of this table.

## Warmup

`Engine.warmup` compiles every kernel variant serving reaches: prefill's page allocation loops over
the batch's largest need, the conv kernel does not specialize on the token count, one batch runs per
row block. With an empty Triton cache no kernel compiles inside a run: 161.68 img/s on LFM3, 66.63
on 4B.

## Arena reuse

Reusing a staging arena waits on an event behind its last queued copy. 9B captions in process, each
of cards 0-3 once per side, two rounds: 46.65 img/s without the wait, 45.38 with it, per card
-8.5%, -0.3%, -2.6%, +5.8%. The gap is inside the run-to-run spread.

## The full GBC graph

`configs/machine/qwen35_card.py` with `mid.py` and `sam3.py` in KohakuGBC: 100 images, one card per
model, the VLM through an engine host with SAM 3 and EmbeddingGemma beside it, grammar replies.

| VLM | Images ok | Engine calls | VLM requests | Seconds | img/s |
|---|---|---|---|---|---|
| LFM3-VL 2B MXFP8 | 100 / 100 | 21,091 | 1,500 | 43.2 | 2.32 |
| Gemma 4 E4B NVFP4 | 100 / 100 | 21,246 | 1,493 | 66.5 | 1.50 |
| Qwen3.5 4B MXFP8 | 100 / 100 | 21,724 | 1,527 | 86.3 | 1.16 |
| Qwen3.5 9B NVFP4 | 100 / 100 | 22,501 | 1,548 | 95.4 | 1.05 |

A 100-image run is mostly ramp and drain: these witness the path, not sustained throughput.

1200 COCO val2017 images, long side 640 px at most, median 0.27 Mpx, under `max_side` 1568, which
never applies to them; 96 resident jobs (the configuration's default):

| VLM | Images ok | Engine calls | VLM requests | Seconds | img/s |
|---|---|---|---|---|---|
| LFM3-VL 2B MXFP8 | 1200 / 1200 | 211,605 | 15,390 | 341.8 | 3.51 |
| Gemma 4 E4B NVFP4 | 1200 / 1200 | 232,896 | 15,838 | 658.7 | 1.82 |
| Qwen3.5 4B MXFP8 | 1200 / 1200 | 242,619 | 16,606 | 706.4 | 1.70 |
| Qwen3.5 9B NVFP4 | 1200 / 1200 | 250,964 | 17,093 | 717.1 | 1.67 |

4B and 9B on a CUDA stream of their own (`own_stream`): 1.68 and 1.67 img/s; a second baseline run
of 4B gave 1.65. The stream does not bind them: resident jobs do.

| VLM | Resident jobs | Images ok | Seconds | img/s | VLM in flight | SAM 3 queued + in flight | Job s mean |
|---|---|---|---|---|---|---|---|
| Qwen3.5 4B | 96 | 1200 / 1200 | 726.7 | 1.65 | 104 | 2 | 58 |
| Qwen3.5 4B | 192 | 1200 / 1200 | 593.5 | 2.02 | 210 | 3 | 93 |
| Qwen3.5 4B | 384 | 1200 / 1200 | 452.2 | 2.65 | 321 | 70 | 140 |
| Qwen3.5 4B | 768 | 1200 / 1200 | 452.9 | 2.65 | 409 | 153 | 216 |
| LFM3-VL 2B | 384 | 1200 / 1200 | 254.4 | 4.72 | 280 | 77 | 75 |
| Gemma 4 E4B | 384 | 1200 / 1200 | 529.7 | 2.27 | 287 | 86 | 158 |
| Qwen3.5 9B | 384 | 1200 / 1200 | 539.2 | 2.23 | 307 | 82 | 168 |

From 384 jobs SAM 3 binds the card. `admit_batch` 8 at 96 jobs: 1.62 img/s, level with the default.
`configs/machine/qwen35_card.py` holds 384 jobs per card on Tokenomos.

An engine failure injected mid-run on LFM3 (100 images): the host exits, the machine starts it
again 62 s later, 95 VLM requests retry, 100 of 100 images finish in 107.5 s.

300 COCO images at 384 jobs, for the models that run slower or from full-precision checkpoints:

| VLM | Images ok | Engine calls | Seconds | img/s |
|---|---|---|---|---|
| LFM3 Omni 2B bf16 | 300 / 300 | 52,415 | 90.7 | 3.31 |
| Gemma 4 12B NVFP4 | 300 / 300 | 62,153 | 283.8 | 1.06 |
| Qwen3.8 27B NVFP4 | 300 / 300 | 60,110 | 423.9 | 0.71 |

Gemma 4 31B NVFP4 at 96 jobs: 300 / 300, 60,908 engine calls, 2,013.2 s, 0.15 img/s. Its nine slots
bind it, so 96 resident jobs are enough.

## LFM3 Omni 2B

1024 rows from the bf16 checkpoint, automatic sizing: 77.87 img/s in process (256 slots, 56,971
pages, 13.2 s wall, 2.2 s encode, 5.3 s prefill, 9.3 s decode wait, 28.08 GiB reserved), 59.06 img/s
over HTTP.

## Vision tower format

Gemma 4 towers stay bf16. Quantizing the blocks' linears to NVFP4 at load, activations scaled per
call, on 16 real images at 1536 px (encoder only): E4B 19.9 img/s against 23.0 in bf16, feature
cosine to bf16 mean 0.950, worst token 0.18. 31B does not run: its 4304 MLP width fits no kernel
(CUTLASS rejects it, b12x needs a multiple of 32).

## What sets the slot count

- 31B: a slot's ring costs 800 MiB at 1024 rows, 9 slots in 10.5 GiB of free memory at `kv_reserve`
  2 GiB.
- 12B: a slot's ring costs 320 MiB, 43 slots at `kv_reserve` 4 GiB.
- 27B: the page pool, 2709 pages at `reserve_new` 256.
- `reserve_new` at `new_cap` (every request reserves all of `max_tokens`) caps live rows at the pool
  over 65 pages: 31B 5.9 of 8, 27B 8.87 img/s in process.
