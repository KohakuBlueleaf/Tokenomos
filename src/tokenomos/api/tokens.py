"""Each token's bytes from a ``tokenizers`` tokenizer (byte-level BPE or SentencePiece), for the
grammar store: added / special tokens and ids past the tokenizer get None (never allowed)."""

import re

from tokenizers import Tokenizer

_BYTE_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def byte_decoder() -> dict[str, int]:
    """GPT-2's byte-level alphabet: printable stand-in character -> byte."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    keep += range(ord("®"), ord("ÿ") + 1)
    chars, n = list(keep), 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(keep, chars)}


def token_bytes(tok: Tokenizer, vocab: int) -> list[bytes | None]:
    pieces = [tok.id_to_token(i) for i in range(min(vocab, tok.get_vocab_size()))]
    added = set(tok.get_added_tokens_decoder())
    # by majority: SentencePiece vocabularies can hold a literal "Ġ" piece
    gs = sum(1 for p in pieces if p is not None and "Ġ" in p)
    byte_level = gs > sum(1 for p in pieces if p is not None and "▁" in p)
    decoder = byte_decoder() if byte_level else None
    out: list[bytes | None] = [None] * vocab
    for i, p in enumerate(pieces):
        if p is None or i in added:
            continue
        if decoder is not None:
            if all(c in decoder for c in p):
                out[i] = bytes(decoder[c] for c in p)
        elif m := _BYTE_PIECE.fullmatch(p):
            out[i] = bytes([int(m.group(1), 16)])
        else:
            out[i] = p.replace("▁", " ").encode()
    return out
