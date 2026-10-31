"""A conversation refused after its images landed in the ring gives the slots back: with two slots
and one worker, a leak leaves the next image waiting forever."""

import base64
import io
from concurrent.futures import TimeoutError as FutureTimeout

import pytest
from PIL import Image

from tokenomos import LLM, SamplingParams

pytestmark = [pytest.mark.gpu, pytest.mark.model]


def picture() -> str:
    raw = io.BytesIO()
    Image.effect_noise((256, 256), 64).convert("RGB").save(raw, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(raw.getvalue()).decode()


def conversation() -> list[dict]:
    content = [{"type": "image_url", "image_url": picture()}, {"type": "text", "text": "What?"}]
    return [{"role": "user", "content": content}]


def test_refused_conversations_return_their_ring_slots(models):
    kw = dict(prep_workers=1, max_images=1, ring_slots=2, slots=4, max_len=1024, new_cap=8)
    with LLM(str(models / "lfm3-vl-2b"), **kw) as llm:
        for _ in range(6):
            bad = SamplingParams(max_tokens=4, json_schema="{ not json")
            with pytest.raises(ValueError):
                llm.submit_chat(conversation(), bad).result(timeout=60)
        fut = llm.submit_chat(conversation(), SamplingParams(max_tokens=4))
        try:
            req = fut.result(timeout=120)
        except FutureTimeout:
            pytest.fail("the ring lost its slots")
        assert llm.output(req).token_ids
