# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact incremental tokenization of chat prompts.

A multi-turn conversation is re-rendered and re-tokenized in full on every
turn, so tokenization cost grows with the conversation (tens of ms per request
at 50K+ tokens). Chat templates such as ChatML start every message with a
special token (``<|im_start|>``). Fast tokenizers split added tokens out of the
text before normalization and pre-tokenization, so cutting the text right
before each occurrence of such a token and tokenizing the pieces separately
gives exactly the token ids of the whole text. Pieces repeat from turn to turn
and are cached; only new messages are tokenized.
"""

import threading
from collections import OrderedDict
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Message delimiters of supported chat templates.
_DELIMITERS = ("<|im_start|>",)


class SegmentTokenCache:
    def __init__(self, tokenizer: Any, max_cached_tokens: int = 32_000_000):
        self.tokenizer = tokenizer
        self.delimiter = self._find_delimiter(tokenizer)
        self.max_cached_tokens = max_cached_tokens
        self._cache: OrderedDict[str, list[int]] = OrderedDict()
        self._cached_tokens = 0
        self._lock = threading.Lock()
        if self.delimiter is not None:
            logger.info(
                "Segment token cache enabled (delimiter %r)", self.delimiter
            )

    @staticmethod
    def _find_delimiter(tokenizer: Any) -> str | None:
        decoder = getattr(tokenizer, "added_tokens_decoder", None)
        if not decoder or not getattr(tokenizer, "is_fast", False):
            return None
        for tok in decoder.values():
            if tok.content not in _DELIMITERS or not tok.special:
                continue
            # The split is exact only if the token does not absorb adjacent
            # whitespace and matches the raw text.
            if tok.lstrip or tok.rstrip or tok.single_word or tok.normalized:
                return None
            return tok.content
        return None

    @property
    def enabled(self) -> bool:
        return self.delimiter is not None

    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` as ``tokenizer(text, add_special_tokens=False)``."""
        d = self.delimiter
        assert d is not None
        segments = []
        start = 0
        while (i := text.find(d, start + 1)) != -1:
            segments.append(text[start:i])
            start = i
        segments.append(text[start:])

        with self._lock:
            found = [self._cache.get(s) for s in segments]
            for s, ids in zip(segments, found):
                if ids is not None:
                    self._cache.move_to_end(s)
        missing = [s for s, ids in zip(segments, found) if ids is None]
        if missing:
            new = dict(
                zip(
                    missing,
                    self.tokenizer(missing, add_special_tokens=False)["input_ids"],
                )
            )
            found = [new[s] if ids is None else ids for s, ids in zip(segments, found)]
            with self._lock:
                for s, ids in new.items():
                    if s not in self._cache:
                        self._cache[s] = ids
                        self._cached_tokens += len(ids)
                while self._cached_tokens > self.max_cached_tokens and self._cache:
                    _, ids = self._cache.popitem(last=False)
                    self._cached_tokens -= len(ids)
        return [t for ids in found for t in ids]
