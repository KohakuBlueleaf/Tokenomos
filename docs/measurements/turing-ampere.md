# RTX 3090 and RTX 2080 Ti

Tensor-core GEMMs from TileLang 0.1.15, kernels compiled with CUDA 13.0; cuBLAS through torch.
Trillion multiply-accumulates per second, best tile of those that compile:

| Shape (M, N, K) | Card | int8, i32 | fp16, f32 | cuBLAS fp16 | cuBLAS bf16 | cuBLAS fp32 |
|---|---|---|---|---|---|---|
| 128, 4096, 2048 | 3090 | 50.1 | 30.7 | 34.3 | 34.5 | 7.5 |
| 256, 4096, 2048 | 3090 | 72.1 | 37.7 | 32.2 | 32.4 | 9.4 |
| 1024, 2048, 2048 | 3090 | 76.3 | 38.5 | 49.1 | 46.3 | 10.5 |
| 256, 65536, 2048 | 3090 | 72.8 | 46.4 | 51.5 | 51.4 | 10.1 |
| 128, 4096, 2048 | 2080 Ti | 17.1 | 11.0 | 20.1 | | 4.5 |
| 256, 4096, 2048 | 2080 Ti | 24.8 | 13.4 | 22.9 | | 5.2 |
| 1024, 2048, 2048 | 2080 Ti | 24.3 | 11.4 | 35.3 | | 5.3 |
| 256, 65536, 2048 | 2080 Ti | 24.9 | 11.8 | 37.0 | | 5.6 |

Int8 products equal the integer product exactly. On the 2080 Ti the `sm_70` wmma route fails to
compile, and tiles past 64 KiB of shared memory are refused.

## Whole-model rates

`bench/suite/run.py --box p8`. Decode: S x 64-token prompts, 512 generated. Prefill: S x 2048-token
prompts. Host: ms of the loop's own work per step. Captions: the 1024 root-caption rows in process,
`max_len` 4096, automatic sizing. S 128 on the 3090, 64 on the 2080 Ti. W8A8 rotates by 16; 4B on
W8A8 carries the int8 GDN state. Peak reserved GiB is the caption run's. Tests: the 3090 322 passed,
97 skipped; the 2080 Ti 218 passed, 201 skipped.

| Model | Work | Card | Decode tok/s | Step ms | Host ms | Prefill tok/s | Prefill host s | img/s | Slots | GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM3-VL 2B | bf16 | 3090 | 15,863 | 8.07 | 2.46 | 31,282 | 0.64 | 47.18 | 256 | 19.06 |
| LFM3-VL 2B | fp16 | 3090 | 17,746 | 7.21 | 3.10 | 32,808 | 0.66 | 48.12 | 256 | 19.71 |
| LFM3-VL 2B | fp32, cuBLAS | 3090 | 13,985 | 9.15 | 3.00 | 24,635 | 4.58 | 35.33 | 256 | 22.79 |
| LFM3-VL 2B | fp32, TileLang | 3090 | 12,054 | 10.62 | 3.34 | 20,083 | 0.82 | 30.65 | 256 | 22.48 |
| LFM3-VL 2B | W8A8 | 3090 | 16,977 | 7.54 | 2.45 | 28,630 | 1.26 | 37.56 | 256 | 17.30 |
| Qwen3.5 4B | bf16 | 3090 | 4,113 | 31.12 | 6.31 | 13,019 | 14.70 | 13.55 | 96 | 20.12 |
| Qwen3.5 4B | fp16 | 3090 | 4,187 | 30.57 | 6.48 | 12,664 | 17.09 | 13.48 | 92 | 20.31 |
| Qwen3.5 4B | fp16, FLA prefill | 3090 | 4,207 | 30.43 | 6.01 | 12,623 | 19.82 | 12.75 | 92 | 19.55 |
| Qwen3.5 4B | fp32, cuBLAS | 3090 | 3,285 | 28.01 | 3.64 | 10,260 | 25.34 | 9.27 | 57 | 21.54 |
| Qwen3.5 4B | fp32, TileLang | 3090 | 2,950 | 30.85 | 4.21 | 8,454 | 23.54 | 8.37 | 52 | 21.21 |
| Qwen3.5 4B | W8A8 | 3090 | 5,973 | 21.43 | 6.33 | 11,361 | 22.89 | 15.68 | 218 | 21.48 |
| Gemma 4 E4B | bf16 | 3090 | 6,423 | 19.93 | 5.26 | 11,104 | 19.90 | 14.28 | 132 | 22.56 |
| Gemma 4 E4B | fp16 | 3090 | 6,658 | 19.23 | 5.91 | 10,397 | 21.38 | 14.48 | 120 | 21.74 |
| Gemma 4 E4B | W8A8 | 3090 | 5,773 | 22.17 | 6.93 | 9,384 | 27.73 | 13.85 | 181 | 21.64 |
| LFM3-VL 2B | fp16 | 2080 Ti | 4,983 | 12.84 | 2.18 | 10,890 | 0.33 | 13.06 | 256 | 18.92 |
| LFM3-VL 2B | fp32, cuBLAS | 2080 Ti | 4,564 | 14.02 | 2.63 | 9,382 | 4.26 | 12.04 | 256 | 20.98 |
| LFM3-VL 2B | fp32, TileLang | 2080 Ti | 3,465 | 18.47 | 5.08 | 5,508 | 0.42 | 7.55 | 256 | 19.95 |
| LFM3-VL 2B | W8A8 | 2080 Ti | 4,338 | 14.75 | 3.50 | 7,476 | 5.69 | 10.96 | 256 | 17.58 |
| Qwen3.5 4B | fp16 | 2080 Ti | 1,500 | 42.68 | 6.98 | 3,464 | 24.66 | 3.40 | 79 | 18.15 |
| Qwen3.5 4B | fp16, FLA prefill | 2080 Ti | 1,498 | 42.72 | 7.53 | 3,333 | 33.31 | 3.20 | 79 | 18.08 |
| Qwen3.5 4B | fp32, cuBLAS | 2080 Ti | 1,426 | 44.90 | 7.78 | 3,270 | 39.46 | 2.85 | 47 | 20.03 |
| Qwen3.5 4B | fp32, TileLang | 2080 Ti | 1,163 | 55.05 | 14.41 | 1,896 | 62.56 | 2.02 | 43 | 19.04 |
| Qwen3.5 4B | W8A8 | 2080 Ti | 1,712 | 37.39 | 11.05 | 2,532 | 51.13 | 3.13 | 175 | 18.86 |
| Gemma 4 E4B | fp16 | 2080 Ti | 1,307 | 48.98 | 5.56 | 2,600 | 47.95 | 2.39 | 90 | 20.57 |
| Gemma 4 E4B | W8A8 | 2080 Ti | 1,125 | 56.90 | 9.91 | 1,899 | 68.35 | 2.20 | 143 | 20.63 |

Prefill host s covers the launches, which block on a full queue, so on 4B and E4B it is near the
wall.

## The GDN prefill

`kernels/gdn/prefill.py`: one kernel per stage (gram, invert, advance) in `work()`, float32 inside,
tiles from `tables/<gpu>.json` (`bench/profile/gdn_kernel.py --write`). 8 x 1024 tokens, µs a call,
largest output error against a float64 reference:

| Card | Shape | Dtype | Tile | `fused` | FlashInfer | FLA | Error `fused` | Error FlashInfer | Error FLA |
|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | 4B | bf16 | 32, 32, 2, 2 | 420 | 383 | 991 | 0.003318 | 0.003464 | 0.003473 |
| RTX 5090 | 4B | fp16 | 32, 32, 2, 2 | 423 | 388 | 998 | 0.000415 | 0.000442 | 0.000444 |
| RTX 5090 | 27B | bf16 | 32, 32, 2, 2 | 556 | 512 | 1,326 | 0.003317 | 0.003462 | 0.003472 |
| RTX 5090 | 27B | fp16 | 32, 32, 2, 2 | 557 | 544 | 1,336 | 0.000415 | 0.000442 | 0.000444 |
| RTX 4090 | 4B | bf16 | 16, 32, 2, 2 | 488 | | 1,510 | 0.003302 | | 0.003472 |
| RTX 4090 | 27B | bf16 | 16, 32, 2, 2 | 670 | | 2,292 | 0.003302 | | 0.003473 |
| RTX 3090 | 4B | bf16 | 32, 32, 2, 2 | 947 | | 2,056 | 0.003303 | | 0.003474 |
| RTX 3090 | 4B | fp16 | 32, 32, 2, 2 | 971 | | 2,055 | 0.000413 | | 0.000444 |
| RTX 3090 | 27B | bf16 | 32, 32, 2, 2 | 1,344 | | 3,053 | 0.003318 | | 0.003472 |
| RTX 3090 | 27B | fp16 | 32, 32, 2, 2 | 1,341 | | 3,039 | 0.000415 | | 0.000444 |
| RTX 2080 Ti | 4B | fp16 | 16, 32, 4, 2 | 28,754 | | 29,000 | 0.000413 | | 0.000443 |
| RTX 2080 Ti | 27B | fp16 | 16, 32, 4, 2 | 42,001 | | 43,857 | 0.000413 | | 0.000443 |

`fused` stays within 10% of FlashInfer on the call and holds level through a whole prefill on a
5090: 4B 73,371 against 75,784 tok/s, 27B 17,300 against 16,999, 64 x 2048 tokens. It runs where
FlashInfer does not, at 2.1–3.1x FLA on the 3090 and 4090. On the 2080 Ti both kernels take 14x the
3090's time while a GEMM takes 2–3x, and `fused` gains 1–4% on FLA.

## fp16 weights, float32 accumulate

Best of a sweep over `(BM, BN, BK, threads)` against cuBLAS, K 2048, TMAC/s:

| Card | N | M | TileLang fp16, f32 | cuBLAS fp16 | cuBLAS fp32 |
|---|---|---|---|---|---|
| 3090 | 4096 | 32 | 10.41 | 9.48 | 4.18 |
| 3090 | 4096 | 128 | 30.77 | 36.42 | 7.53 |
| 3090 | 4096 | 256 | 39.06 | 31.93 | 9.26 |
| 3090 | 4096 | 1024 | 43.71 | 51.76 | 10.58 |
| 3090 | 65536 | 256 | 47.67 | 50.81 | 9.73 |
| 3090 | 65536 | 1024 | 50.21 | 58.64 | 9.75 |
| 2080 Ti | 4096 | 32 | 5.09 | 6.20 | 2.94 |
| 2080 Ti | 4096 | 128 | 9.89 | 21.16 | 4.50 |
| 2080 Ti | 4096 | 256 | 13.98 | 22.72 | 5.18 |
| 2080 Ti | 4096 | 1024 | 18.55 | 36.41 | 5.33 |
| 2080 Ti | 65536 | 256 | 17.64 | 35.41 | 5.24 |
| 2080 Ti | 65536 | 1024 | 18.62 | 37.65 | 6.05 |

Largest error against a float64 product of the same inputs, 2048 by 4096 columns: TileLang
0.00008-0.00013, cuBLAS fp16 0.0077-0.0142. The 3090 tracks cuBLAS fp16 at 2.5-5x cuBLAS fp32; the
2080 Ti TileLang tiles reach half of cuBLAS fp16 and 2-3x cuBLAS fp32.

## Work dtypes

32 README prompts, 128-row decode of 128 tokens, RTX 3090. `work="fp32"` keeps activations, state and
embeddings in float32 with fp16 weights; logits are on the dense backend, 1,024 tokens of real text
against a float32 Hugging Face forward, log-probability gaps at its top token:

| Model | Work | Route | tok/s, 128 rows | GiB | Argmax | Gap mean | Gap p99 |
|---|---|---|---|---|---|---|---|
| LFM3-VL 2B | bf16 | | 11,893 | 5.71 | 0.98730 | 0.016196 | 0.13261 |
| LFM3-VL 2B | fp16 | | 12,850 | 5.71 | 0.99902 | 0.002085 | 0.01675 |
| LFM3-VL 2B | fp32 | cuBLAS | 10,731 | 8.65 | 0.99902 | 0.001400 | 0.01024 |
| LFM3-VL 2B | fp32 | TileLang | 9,585 | 8.61 | 1.00000 | 0.000817 | 0.00709 |
| LFM3-VL 2B | fp32 + W8A8 | cuBLAS | 11,870 | 7.30 | | | |
| Qwen3.5 4B | bf16 | | 3,456 | 20.40 | 0.98535 | 0.013722 | 0.08058 |
| Qwen3.5 4B | fp16 | | 3,481 | 19.84 | 0.99609 | 0.001762 | 0.01176 |
| Qwen3.5 4B | fp32 | cuBLAS | 2,616 | 20.80 | 0.99707 | 0.001612 | 0.00786 |
| Qwen3.5 4B | fp32 | TileLang | 1,862 | 19.98 | 0.99805 | 0.000508 | 0.00362 |
| Qwen3.5 4B | fp32 + W8A8 | cuBLAS | 3,203 | 19.74 | | | |

fp16 holds bf16's speed at an eighth of its logit gap; against fp16, fp32 costs 16-47% of the rate
and shrinks the mean gap a further 1.1-3.5x. GiB is allocated after load.

## W8A8

Made from a bf16 checkpoint at load: per-row weight scale, per-token activation scale, the decoder
linears only (tower, head and skinny linears stay). Greedy against bf16 on 32 README prompts, 48
tokens, RTX 3090, decode at 128 sequences of 128 tokens:

| Model | Variant | tok/s | First token | Shared prefix | Identical |
|---|---|---|---|---|---|
| LFM3-VL 2B | bf16 | 12,126 | | | |
| LFM3-VL 2B | W8A8 | 13,688 | 0.906 | 32.0 | 19 / 32 |
| LFM3-VL 2B | W8A8, rotate 16 | 12,652 | 0.938 | 38.6 | 25 / 32 |
| Qwen3.5 4B | bf16 | 3,159 | | | |
| Qwen3.5 4B | W8A8 | 3,557 | 0.875 | 29.6 | 15 / 32 |
| Qwen3.5 4B | W8A8, rotate 16 | 3,327 | 1.000 | 37.9 | 20 / 32 |

Rotation groups on LFM3, agreement measured on an unfused build of the same weights: group 4 0.969
and 40.2, group 16 1.000 and 38.1, group 64 0.969 and 38.9, group 256 1.000 and 40.2.

## The full GBC graph

`mid.py` and `sam3.py`, 1200 COCO images on the RTX 3090, SAM 3 and EmbeddingGemma on the card:

| VLM | Resident jobs | Images ok | Seconds | img/s | SAM 3 busy |
|---|---|---|---|---|---|
| LFM3-VL 2B bf16 | 384 | 1200 / 1200 | 747.1 | 1.61 | 1.00 |
| LFM3-VL 2B W8A8, rotate 16 | 384 | 1200 / 1200 | 820.5 | 1.46 | 1.00 |
| LFM3-VL 2B W8A8 | 384 | 1200 / 1200 | 870.3 | 1.38 | 1.00 |

SAM 3 is saturated on this card in every row, so a faster VLM does not raise the rate. Qwen3.5 4B
W8A8 with rotate 16 and the int8 GDN state at 192 jobs finished 899 images in the 1500 s limit, 0.60
img/s.

The RTX 2080 Ti, LFM3-VL 2B, 192 jobs:

| VLM | Images | Images ok | Seconds | img/s | VLM busy | SAM 3 busy |
|---|---|---|---|---|---|---|
| fp16, non-blocking vision graph | 100 | 100 / 100 | 277.2 | 0.36 | | |
| W8A8, rotate 16, SAM 3 vision graph off | 412 of 1200, 1500 s limit | 412 / 412 | 1411.5 | 0.29 | 1.00 | 0.95 |

Both the VLM and SAM 3 fill this card.

## Gemma 4 under fp16

E4B from its bf16 checkpoint on the RTX 2080 Ti, at 64 sequences of 64 + 256 tokens. Attention tiles
are half the size of sm80 and later, with a floor of 16: Turing has 64 KiB of shared memory.

| Decode tok/s | Step ms, median | Host ms per step | Prefill tok/s, 2048-token prompts | Prefill host s |
|---|---|---|---|---|
| 1,120.8 | 57.1 | 3.30 | 2,631.7 | 47.4 of 49.8 |
