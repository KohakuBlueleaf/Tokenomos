# Bottlenecks

One RTX 5090 32 GB per run, cards 2 and 3 of the box, default kernels (`BACKEND` `table`, the tuned
tables, no override). All source tree hash `f14fab5d725b`. Rows: the first 1024 of
`calib_rows.jsonl`, greedy, thinking off, `max_tokens` 1024, 16 prep workers, automatic sizing.
Each image is one JPEG, mean 66.7 KB as a data URL, mean 0.227 Mpix.

| Name | Checkpoint | Arch | Slots | Pages | Prompt tokens | Generated tokens |
|---|---|---|---|---|---|---|
| LFM3 | `lfm3-vl-2b-mxfp8` | `head` mxfp8 | 256 | 60507 | 1028.0 | 83.8 |
| 4B | `qwen35-4b-mlp4-mxfp8` | `head` mxfp8, `state_format` int8 | 256 | 16625 | 1029.6 | 114.5 |
| E4B | `gemma4-e4b-it-nvfp4` | `head` nvfp4 | 256 | 21606 | 1051.9 | 81.5 |
| 9B | `qwen35-9b-nvfp4` | `head` bf16, `state_format` int8 | 256 | 13118 | 1029.6 | 109.7 |
| 12B | `gemma4-12b-it-nvfp4` | `head` nvfp4 | 42 | 2964 | 1055.9 | 92.4 |
| 27B | `qwen38-27b-nvfp4-full` | `head` nvfp4, `state_format` int8 | 80 | 3243 | 1029.6 | 107.2 |
| 31B | `gemma4-31b-it-nvfp4-redhat` | `head` nvfp4 | 9 | 488 | 1055.9 | 95.0 |

Scripts: `bench/prep.py`, `bench/http_front.py`, `bench/caption.py --tap`, `bench/decode.py --tap`,
`bench/vision.py`; `bench/tap.py` holds the timing taps. Reports: `reports/prep_bn_*`,
`http_front_bn`, `caption_bn_*_c2` and `_c3`, `decode_bn_raw_*`, `decode_bn_shape_*`, `vision_bn_*`.
The two-card columns are the mean of one run on card 2 and one on card 3.

GPU seconds are CUDA timing events around each encode, prefill forward and decode graph. Encode and
prefill run eagerly, so their spans include launch gaps between kernels; the saturated rate at a
large batch is the hardware figure.

## Frontend

Milliseconds per request, mean, one thread, 1024 rows. Host load average 38-45 during these runs
(a GBC arm on card 1).

| Model | Render | Tokenize | Base64 | Image decode | Resize | Expand | Shared prefix | Ring write | Task pickle | Result pickle | RoPE or runs | `worker.job` whole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LFM3 | 0.265 | 1.876 | 0.232 | 1.279 | 1.451 | 0.060 | 2.086 | 0.104 | 0.043 | 0.115 | 0 | 7.27 |
| 4B | 0.217 | 2.015 | 0.219 | 1.219 | 1.382 | 0.058 | 0.433 | 0.098 | 0.037 | 0.095 | 0.245 | 5.63 |
| E4B | 0.249 | 1.598 | 0.215 | 1.225 | 2.055 | 0.059 | 1.764 | 0.156 | 0.042 | 0.106 | 0 | 7.20 |
| 9B | 0.218 | 2.031 | 0.223 | 1.233 | 1.392 | 0.059 | 0.432 | 0.099 | 0.037 | 0.095 | 0.247 | 5.67 |
| 12B | 0.256 | 1.688 | 0.218 | 1.268 | 2.094 | 0.060 | 1.814 | 0.183 | 0.043 | 0.109 | 0.089 | 7.40 |
| 27B | 0.220 | 2.013 | 0.220 | 1.221 | 1.378 | 0.058 | 0.453 | 0.100 | 0.036 | 0.095 | 0.245 | 5.65 |
| 31B | 0.253 | 1.629 | 0.215 | 1.224 | 2.062 | 0.059 | 1.784 | 0.165 | 0.042 | 0.108 | 0.088 | 7.25 |

Shared prefix re-renders the conversation before its last message and tokenizes it again. Qwen3.5's
template refuses that partial conversation, so its cost stops at the render. RoPE and runs are built
in the serving process at submit; everything left of them runs in a worker. Image decode's p50 is
half its mean (0.6 ms): a few large images carry it.

Prep pool, whole conversations with 16 workers and the same payloads through a job that does
nothing (IPC alone):

| Model | Rows/s, 8 workers | 16 | 32 | 64 | Serving CPU us per row, 16 workers | IPC alone rows/s, 16 workers | Serving CPU us per row, IPC alone |
|---|---|---|---|---|---|---|---|
| LFM3 | 848 | 1097 | 1121 | 817 | 468 | 4132 | 282 |
| 4B | 1032 | 1313 | 1255 | 567 | 485 | 3866 | 297 |
| E4B | 772 | 971 | 1329 | 417 | 489 | 3806 | 303 |
| 9B | 1032 | 1030 | 1806 | 522 | 476 | 3525 | 330 |
| 12B | 864 | 957 | 758 | 357 | 493 | 3459 | 331 |
| 27B | 1035 | 1031 | 1130 | 903 | 471 | 3489 | 330 |
| 31B | 827 | 935 | 1039 | 448 | 507 | 4041 | 286 |

Each pool pass lasts about a second, on a loaded host; the spread between rows is mostly noise.

HTTP handling alone: the real handler and server class over a stub `LLM` that finishes every request
at once, real bodies (mean 70.6 KB), real reply texts, a connection per request.

| In flight | Requests/s | Server CPU us per request | Latency p50 ms |
|---|---|---|---|
| 1 | 716 | 858 | 0.90 |
| 16 | 1373 | 910 | 10.5 |
| 64 | 1399 | 898 | 44.6 |
| 256 | 1385 | 905 | 180 |
| 1024 | 1389 | 902 | 344 |

Of those 900 us: JSON body parse 185 us, `params` 3 us, reply encode 12 us; the rest is the
connection, its thread and the stdlib's request parsing. The server is one GIL, so 1389 requests/s
is its ceiling. In `captions.md` HTTP costs LFM3 0.81 ms per image over in-process, the same size.

Engine loop host work on the caption rows:

| Model | Loop outside admission, ms per step | `_read`, ms per step | Replay, ms per step | Encode launch, ms per image | Submit in serving process, ms per image | GPU uncovered s |
|---|---|---|---|---|---|---|
| LFM3 | 0.78 | 0.21 | 0.03 | 0.89 | 0.10 | 0.19 |
| 4B | 0.50 | 0.15 | 0.03 | 1.17 | 0.55 | 0.21 |
| E4B | 1.33 | 0.31 | 0.43 | 1.82 | 0.10 | 0.19 |
| 9B | 2.64 | 0.18 | 2.44 / 1.71 | 1.66 | 0.56 | 0.18 |
| 12B | 0.27 | 0.08 | 0.02 | 0.45 | 0.21 | 0.82 |
| 27B | 0.34 | 0.11 | 0.03 | 2.65 | 0.47 | 0.41 |
| 31B | 0.18 | 0.04 | 0.02 | 21.9 | 0.19 | 0.23 |

GPU uncovered is wall minus every timed span: the GPU's idle time plus short copies, 0.1-2.9 % of
the wall. It bounds what host work costs the card. Engine-thread CPU seconds equal wall seconds on
every model because CUDA waits spin, so CPU time cannot split host work from waiting. Admission host
seconds (`admit_host_s`) run at 59-103 % of the encode and prefill GPU seconds. 9B's replay holds
the host 1.7-2.4 ms a step against 0.03 ms on 4B; the GPU does not idle for it.

Starvation: a decode step with free slots, nothing queued and prepared requests still to come. 0
steps on all 14 runs. The last request enters the engine at 75-87 % of the wall: the ring (128
slots) holds workers back, the engine is never waiting on them.

Ranking by rate: prep pool at 16 workers 935-1313 rows/s, HTTP 1389 requests/s (GIL), pool IPC
3459-4132 rows/s. Ranking by cost per request: shared prefix on LFM3 (2.09 ms), tokenize on Qwen
(2.0 ms), resize on Gemma (2.06 ms), then HTTP (0.90 ms). The whole frontend sits at 6x or more of
the fastest model's caption rate.

## Side costs per model

Caption rows, seconds on the card, shares of the wall in parentheses. Frontend runs beside the card
in workers: CPU seconds over 1024 rows, 0 s starved.

| Model | img/s, card 2 / 3 | Wall s | Frontend CPU s | Vision GPU s | Prefill GPU s | Decode GPU s | Prefill tok/s | Decode tok/s | Host (GPU uncovered) s | Largest |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM3 | 155.23 / 156.23 | 6.6 | 7.4 | 2.22 (34 %) | 2.40 (36 %) | 1.76 (27 %) | 147,900 | 48,800 | 0.19 | prefill, vision |
| 4B | 60.74 / 59.54 | 17.0 | 5.8 | 4.42 (26 %) | 4.85 (28 %) | 7.55 (44 %) | 72,600 | 15,500 | 0.21 | decode |
| E4B | 58.16 / 57.96 | 17.6 | 7.4 | 7.90 (45 %) | 4.85 (27 %) | 4.70 (27 %) | 79,700 | 17,800 | 0.19 | vision |
| 9B | 49.18 / 44.78 | 21.8 | 5.8 | 7.67 (35 %) | 5.74 (26 %) | 8.27 (38 %) | 61,700 | 13,800 | 0.18 | decode, vision |
| 12B | 22.16 / 23.12 | 45.2 | 7.6 | 0.21 (0.5 %) | 11.81 (26 %) | 32.40 (72 %) | 32,600 | 2,920 | 0.82 | decode |
| 27B | 17.68 / 19.49 | 55.2 | 5.8 | 7.96 (14 %) | 18.70 (34 %) | 28.17 (51 %) | 18,800 | 3,930 | 0.41 | decode |
| 31B | 3.59 / 3.63 | 283.7 | 7.4 | 27.61 (10 %) | 74.80 (26 %) | 181.07 (64 %) | 14,460 | 537 | 0.23 | decode |

Decode tok/s here: generated tokens over decode GPU seconds, the two runs averaged. Prefill tokens
per image: 346, 344, 377, 345, 376, 343 and 1056; 31B holds 9 slots, under the 16 where the prefix
table starts, so it prefills the whole prompt. 12B's vision is one linear patch embedder, no tower.
`captions.md`'s Decode wait is the host's wait on an event and includes queued encode and prefill
work; decode GPU seconds above are the time the graphs ran.

A runaway reply sits alone in the smallest bucket until `max_tokens` or a late stop, at a full step
cost:

| Model | Steps in the smallest bucket, normal run | With a long row | Decode GPU s in it | img/s without / with |
|---|---|---|---|---|
| 4B | no run without | 458, 496 | 1.68, 1.83 | |
| 9B | 88 | 457 | 2.25 | 49.18 / 44.78 |
| 27B | 32 | 460 | 5.18 | 19.49 / 17.68 |
| LFM3 | 26, 90 | 788 in a run with one runaway reply | 1.6 at 2.1 ms | 155.7 / 129.35 |

That row accounts for most of the 9B and 27B spreads. `captions.md`'s LFM3 run, on an earlier source
tree, carried one; without those 1.6 s its wall matches the runs here.

Raw decode, S × 64-token prompts, 512 generated, one card:

| Model | Slots | Requests | Wall s | Prefill GPU s | Decode GPU s | Host (GPU uncovered) s | Host ms per step | Decode tok/s whole call | Largest |
|---|---|---|---|---|---|---|---|---|---|
| LFM3 | 256 | 256 | 2.06 | 0.15 (7 %) | 1.90 (92 %) | 0.013 | 0.43 | 63,522 | decode |
| 4B | 256 | 256 | 5.67 | 0.38 (7 %) | 5.29 (93 %) | 0.007 | 0.72 | 23,100 | decode |
| 4B | 512 | 512 | 10.55 | 0.68 (6 %) | 9.86 (93 %) | 0.009 | 1.55 | 24,856 | decode |
| E4B | 256 | 256 | 5.31 | 0.22 (4 %) | 5.08 (96 %) | 0.008 | 0.63 | 24,688 | decode |
| 9B | 256 | 256 | 7.02 | 0.39 (6 %) | 6.62 (94 %) | 0.007 | 0.84 | 18,678 | decode |
| 12B | 45 | 128 | 13.31 | 0.32 (2 %) | 12.98 (98 %) | 0.011 | 0.35 | 4,925 | decode |
| 27B | 80 | 128 | 15.91 | 0.66 (4 %) | 15.25 (96 %) | 0.007 | 0.75 | 4,120 | decode |
| 31B | 9 | 32 | 27.52 | 0.43 (2 %) | 27.08 (98 %) | 0.012 | 0.37 | 595 | decode |

No frontend and no vision on this path.

## Card ceilings

Each stage alone at its best measured rate. Vision: images/s at the best batch of the sweep
(`bench/vision.py`, 512 images, batches 1-256 under a 0.9 memory cap) or the caption runs' spans,
the higher. Prefill: the highest GPU rate of the 2048-token run, the caption-shaped run, the caption
runs and the docs (`gemma4.md`, `lfm3-vl-2b.md`). Decode: rows over the median GPU step with every
slot live, 128 generated.

| Model | Vision img/s | Best batch | Batch the caption runs used | Prefill tok/s | Decode tok/s at slots, caption context | Decode tok/s at slots, 64-token prompts | Step ms, caption context / short | Doc decode peak tok/s |
|---|---|---|---|---|---|---|---|---|
| LFM3 | 471 | 128 | 22 | 155,772 | 50,078 | 68,248 | 5.11 / 3.75 | 79,030 (512 rows) |
| 4B | 232 | 16 | 22 | 79,706 | 18,554 | 24,780 | 13.80 / 10.33 | 24,856 (512 rows, here) |
| E4B | 140 | 4 | 21 | 85,905 | 16,786 | 25,703 | 15.25 / 9.96 | 29,524 (512 rows) |
| 9B | 134 | 16 | 15 | 64,062 | 15,651 | 19,860 | 16.36 / 12.89 | 18,825 (256 rows) |
| 12B | 4,810 | 16 | 5 | 37,401 | 3,053 | 5,235 | 14.74 / 8.60 | 7,948 (104 rows) |
| 27B | 129 | 16 | 7 | 18,984 | 4,435 | 4,987 | 18.04 / 16.04 | 4,337 (64 rows) |
| 31B | 42 | 32 | 1 | 14,493 | 548 | 675 | 16.43 / 13.34 | 1,109 (18 rows) |

Caption context means prompts of the model's mean length; the step sits 12-71 % over the 64-token
step, so raw decode rates overstate the caption ceiling. 12B ran these at 45 slots. The doc peaks
are whole-call rates at the sequence count named.

Against the caption workload, per image, each stage's ceiling time, and the sum as the card's rate:

| Model | Vision ms | Prefill ms | Decode ms | Ceiling img/s | Measured img/s | Measured / ceiling | Ceiling stage | Stage alone img/s: vision, prefill, decode |
|---|---|---|---|---|---|---|---|---|
| LFM3 | 2.12 | 2.22 | 1.67 | 166 | 155.7 | 94 % | prefill, vision | 471, 450, 598 |
| 4B | 4.30 | 4.32 | 6.17 | 67.6 | 60.1 | 89 % | decode | 232, 232, 162 |
| E4B | 7.17 | 4.39 | 4.85 | 60.9 | 58.1 | 95 % | vision | 140, 228, 206 |
| 9B | 7.44 | 5.39 | 7.01 | 50.4 | 47.0 | 93 % | vision, decode | 134, 186, 143 |
| 12B | 0.21 | 10.06 | 30.26 | 24.7 | 22.6 | 92 % | decode | 4810, 99, 33 |
| 27B | 7.73 | 18.09 | 24.17 | 20.0 | 18.6 | 93 % | decode | 129, 55, 41 |
| 31B | 23.75 | 72.86 | 173.30 | 3.70 | 3.61 | 98 % | decode | 42, 14, 5.8 |

The stages share one card, so they add. The gap to the ceiling is decode buckets paid for and part
empty, the tail above, and vision below its best batch. The decode ceiling is not strict: E4B's
caption rate (17,800) and one LFM3 run (51,000) pass it, because rows that have stopped leave a
step cheaper than the all-live step. On 12B, 27B and 31B the slots cap decode: step cost grows far
slower than rows (bucket 8 to full: 31B 14.2 to 16.4 ms, 27B 10.9 to 18.0 ms, 12B 6.3 to 14.7 ms),
and a slot's memory sets the count (`captions.md`).

Vision past its best batch: flat within 2 % on LFM3, E4B 139 to 130 img/s from batch 4 to 128, 9B
122 to 113 from 16 to 256. At batch 1 a tower gives 24-87 % of its best (LFM3 110 of 454, 31B 36.5
of 42.1); 31B's caption runs admit one image at a time and sit at 37.

The vision sweep's best rate falls under the caption spans on 4B and 9B (217 and 122 against 232 and
134): the sweep's 512 images and the runs' 1024 differ, so the table uses the higher.
