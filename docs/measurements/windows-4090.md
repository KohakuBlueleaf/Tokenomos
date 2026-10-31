# RTX 4090 on Windows

Windows 11, RTX 4090 sm89, Python 3.12, torch 2.14.1+cu130, `triton-windows` 3.9.0.post30, TileLang
0.1.15 compiling through nvcc and `cl.exe`, no FlashInfer. Every run is capped at 19 GiB of the
card's 24 (`torch.cuda.set_per_process_memory_fraction`, `bench/suite/capped.py`), `kv_leave` 1.5 GiB,
`kv_reserve` 5 GiB, `head` bf16, `gdn_prefill` `fused` unless a row says otherwise. Tests: 359 passed,
60 skipped.

Decode: S x 64-token prompts, 512 generated, random prompts, greedy, EOS ignored. Prefill: 32 x
2048-token prompts, 1 generated. Host: ms of the loop's own work per step at the largest S that ran.
Captions: 1024 GBC root-caption rows, greedy, thinking off, `max_tokens` 1024, `max_len` 4096.

| Model | Linears | GDN state | Decode tok/s S32 | S128 | S256 | Prefill tok/s | Host ms | Captions in process | Captions HTTP | Slots |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM3-VL 2B | bf16 | | 6,235 | 20,706 | 32,579 | 47,558 | 1.89 | 71.07 | 68.19 | 256 |
| LFM3-VL 2B | W8A8, rotate 16 | | 7,887 | 21,297 | 34,011 | 51,108 | 1.78 | 71.95 | 71.23 | 256 |
| LFM3-VL 2B | FP8 | | 7,077 | 19,498 | 30,259 | 59,103 | 1.52 | 79.40 | | 256 |
| Qwen3.5 4B | bf16, FLA prefill | int8 | 2,734 | 7,692 | 9,853 | 17,962 | 10.09 | 22.42 | | 127 |
| Qwen3.5 4B | bf16 | int8 | 2,777 | 7,699 | 9,642 | 18,548 | 9.36 | 22.40 | | 127 |
| Qwen3.5 4B | W8A8, rotate 16 | int8 | 3,105 | 7,769 | 10,415 | 21,121 | 8.12 | 23.93 | 24.35 | 176 |
| Qwen3.5 4B | FP8 | int8 | 2,873 | 7,366 | 10,316 | 22,721 | 8.21 | 25.73 | | 135 |
| Qwen3.5 4B | FP8, GDN projections too | int8 | 3,049 | 7,626 | 10,656 | 24,305 | 5.57 | 27.45 | | 158 |
| Qwen3.5 4B | FP8, GDN projections too | fp8 | 3,181 | 7,838 | 10,301 | 23,814 | 5.48 | 27.09 | | 158 |
| Qwen3.5 9B | RedHatAI FP8 dynamic | int8 | 1,860 | 5,126 | 4,155 | 11,347 | 4.40 | 9.50 | 9.44 | 29 |
| Qwen3.5 9B | RedHatAI FP8 dynamic | fp8 | 1,948 | 5,559 | 4,449 | 11,173 | 4.17 | 9.51 | | 29 |
| Gemma 4 E4B | bf16 | | 2,576 | | | 16,005 | 2.98 | 15.75 | | 36 |
| Gemma 4 E4B | W8A8, rotate 16 | | 2,602 | 7,539 | | 18,662 | 2.67 | 21.66 | | 105 |

- 9B at S256 runs 96 rows: that is what the 19 GiB pool holds, so its S256 column is lower than
  S128.
- E4B bf16 at S64: 4,696 tok/s, 13.63 ms a step; W8A8 4,495, 14.24 ms. E4B runs out of the cap at
  S128 in bf16 and at S256 in W8A8: a dense ring costs each slot its window.
- MXFP8 and NVFP4 do not run on sm89: cuBLAS answers `NOT_SUPPORTED` for MXFP8, ptxas rejects the
  NVFP4 kernels.

## The GDN prefill

Eight 1024-token sequences through one chunked GDN call at 4B's shape, against a float64 reference
of the same inputs:

| Backend | Dtype | µs | Largest error |
|---|---|---|---|
| FLA | bf16 | 1,444 | 0.003472 |
| `fused` | bf16 | 486 | 0.003302 |
| FLA | fp16 | | 0.000444 |
| `fused` | fp16 | | 0.000413 |

`fused` is 3x FLA on the call. A GDN call is about a tenth of a layer's prefill on 4B, so a whole
forward gains 3%: 18,548 against 17,962 tok/s.

## FP8 formats

`float-quantized` compressed-tensors checkpoints hold E4M3 weights with scales per output row,
per channel or per 128 x 128 block. `kernels/linear/fp8.py` reads each of these as a scale per row
per 128 columns, scales each activation row per 128 columns from its own amax and multiplies on
Triton FP8 tensor-core dots with float32 accumulation.
