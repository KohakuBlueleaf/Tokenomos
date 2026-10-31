# JSON grammar on the device: measured

LFM3-VL 2B (MXFP8, MXFP8 head) on one RTX 5090. Schemas: the 7 GBC call schemas of
`KohakuGBC/poc/gbc-poc/grammars/json` (image, entity, entity_batch, entity_leafcap, relation,
composition, seed_elements), round-robin over GBC root-caption chats, greedy, 1024 max tokens,
whitespace knob 8. (`bench/grammar.py`, `bench/decode.py --grammar`, `bench/profile/nsys_steps.py`)

## Correctness

1024 replies (cold 256, then 512, then warm 256, one table throughout):

| check | result |
|---|---|
| mask decisions compared with the standalone Rust matcher (every output token) | 104,730 |
| mask disagreements | 0 |
| rejected transitions | 0 |
| finished replies that fail `jsonschema` validation | 0 |
| replies cut by max tokens | 0 |

## Cost per decode step (GPU, nsys)

`off`: engine without grammar; `idle`: grammar kernels in the step, rows free text; `on`: every
row on the image schema (rows that finish their JSON keep decoding under the end-of-sequence
mask). 512 generated tokens per row.

| seqs | busy ms off | grammar ms idle | grammar ms on | share on |
|---|---|---|---|---|
| 64 | 2.110 | 0.003 | 0.007 | 0.4 % |
| 256 | 3.934 | 0.011 | 0.027 | 0.7 % |

## Misses (rows paused one step while the host adds the entry)

| phase | replies | misses | of tokens | host s resolving | states | distinct masks |
|---|---|---|---|---|---|---|
| cold (empty table) | 256 | 6,682 | ~25 % | 0.85 | 2,962 | 323 |
| after 256 | 512 | 1,842 | ~3.5 % | 0.50 | 4,112 | 384 |
| warm (after 768) | 256 | 828 | ~3 % | 0.35 | 4,735 | 410 |

Warm misses: 511 token entries (string ends, keys, enums under a new parent counter) and 317
(shape, counters) entries; states still grow slowly with the array-index combinations.
