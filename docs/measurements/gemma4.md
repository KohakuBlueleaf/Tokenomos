# Gemma 4

One RTX 5090 32 GB per run; Gemma rows use the same card for Tokenomos and vLLM. Random-token
prompts, greedy, EOS ignored. Tokenomos: CUDA graphs, tuned tables, engine warmup, NVFP4 or MXFP8
head, prefix table off. Stock vLLM 0.32.0: no plugin, bf16 head, 0.90 of the card, `LLM` class,
prefix caching off.

## Decode, whole-run peak

64-token prompts, 512 generated. Generated tok/s over the whole call.

| Model | Tokenomos | at sequences | rows held | vLLM | at sequences |
|---|---|---|---|---|---|
| E4B NVFP4 | 29,524 | 512 | 512 | 17,462 | 256 |
| E4B MXFP8 | 23,366 | 512 | 512 | 16,809 | 256 |
| 12B NVFP4 | 7,948 | 512 | 104 | 6,392 | 256 |
| 31B NVFP4 (Red Hat) | 1,109 | 32 | 18 | 1,042 | 64 |

## Decode by sequence count

| Model, sequences | 8 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| E4B NVFP4 Tokenomos | 2,136 | 8,020 | 13,837 | 20,346 | 25,544 | 29,524 |
| E4B NVFP4 vLLM | 1,832 | 6,432 | 10,900 | 15,266 | 17,462 | 17,213 |
| E4B MXFP8 Tokenomos | 1,594 | 5,966 | 10,643 | 16,241 | 20,720 | 23,366 |
| E4B MXFP8 vLLM | 1,489 | 5,301 | 9,320 | 14,243 | 16,809 | 15,545 |
| 12B NVFP4 Tokenomos | 1,315 | 4,153 | 6,381 | 6,320 | 7,348 | 7,948 |
| 12B NVFP4 vLLM | 1,061 | 3,539 | 5,494 | 6,302 | 6,392 | 5,987 |
| 31B NVFP4 Tokenomos | 610 | 1,109 | 1,107 | 1,108 | | |
| 31B NVFP4 vLLM | 421 | 1,020 | 1,042 | 1,004 | | |

## Prefill

2048-token prompts, 1 generated. Prompt tok/s, best sequence count.

| Model | Tokenomos | vLLM |
|---|---|---|
| E4B NVFP4 | 83,750 | 71,234 |
| E4B MXFP8 | 51,707 | 46,486 |
| 12B NVFP4 | 33,931 | 28,410 |
| 31B NVFP4 | 13,425 | 11,605 |

## Choices

- Head: NVFP4 quantized at load. Agreement with the bf16 head stays within noise on E4B (argmax
  0.8475 and 0.8836 against 0.8456 and 0.8859) and 12B (0.8347 and 0.8592 against 0.8363 and
  0.8749). Qwen3.5 loses agreement (9B argmax 0.9165 to 0.9043), so it stays bf16.
- Tiles: head dim 256 runs (32, 32), dim 512 runs (16, 16). Sizes are powers of two; (16, 8) does
  not compile at dim 512.
- Rows: a sliding ring costs 800 MiB a slot on 31B and 320 MiB on 12B at 1024 rows, which sets their
  slot counts; a sequence that fits a shorter `max_len` costs less.
- Warmup runs every kernel variant first: b12x compiles a CuTe kernel on a tile's first use.
- Captions: `captions.md`.
