"""Deterministic cleaning of PDF-parser output (MinerU markdown).

Used by schema.build_messages(), i.e. identically at training and inference time.
Only lossless-for-the-task changes are made: HTML tables, <sup>/<sub> and LaTeX are
kept because trait values live in them.
"""

from __future__ import annotations

import re
import unicodedata

# C0/C1 control characters except TAB (\x09) and LF (\x0a); CR is normalised below.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Zero-width characters that only waste tokens or break word matching.
_ZERO_WIDTH = re.compile(r"[​-‍⁠﻿]")
# ![alt](images/page_0_image_9.jpg): the image itself is not available to a text model.
# The alt text cannot contain "[", and the path cannot contain "(". For this reason, each search
# stops at the next "[" or "(", and the time is linear also for hostile input.
_IMAGE = re.compile(r"!\[([^\[\]]*)\]\([^()]*\)")
_BLANK_RUNS = re.compile(r"\n{3,}")


def clean_markdown(text: str, *, drop_images: bool = True) -> str:
    """Normalise Unicode/newlines, drop image links (keep alt text), squeeze blank lines."""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _ZERO_WIDTH.sub("", _CONTROL.sub("", t))
    if drop_images:
        t = _IMAGE.sub(lambda m: m.group(1).strip(), t)
    # Remove the spaces and TABs at the end of each line. Do not use a regex such as [ \t]+\n:
    # its time is quadratic on a long run of spaces without a line break.
    t = "\n".join(line.rstrip(" \t") for line in t.split("\n"))
    t = _BLANK_RUNS.sub("\n\n", t)
    return t.strip()
