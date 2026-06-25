# Copyright 2026 — SOD benchmark (AIME-TIR) integration for SDAR. Apache-2.0.
"""Action projection for the math_tool (AIME-TIR) environment.

Unlike the search env (which extracts a single ``<search>`` / ``<answer>``
block), the math_tool env keeps the *full* action text: the per-sample
``MathToolEnv`` parses the action itself (it must distinguish a code tool call
from a final boxed answer, and a tool call may contain arbitrary multi-line
Python that we should not mangle with regex extraction). So this projection is a
near pass-through; it only computes a ``valid`` flag.

An action is considered *valid* (``1``) when it contains at least one of:
    * a hermes-style ``<tool_call>...</tool_call>`` block,
    * a fenced ```python ... ``` (or generic ``` ``` ```) code block, or
    * a ``\\boxed{...}`` final answer.
Otherwise it is ``0`` (the model produced neither a tool call nor an answer).
"""
from typing import List, Tuple
import re

# Pre-compiled detection patterns (kept loose; the env does the precise parsing).
_RE_TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.IGNORECASE | re.DOTALL)
_RE_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_RE_BOXED = re.compile(r"\\boxed\s*{", re.IGNORECASE)


def math_tool_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project raw LLM actions for the math_tool env.

    Args:
        actions: list of raw decoded model action strings (one per env).

    Returns:
        (results, valids): ``results`` is the action text passed through
        unchanged (the env parses it); ``valids[i]`` is ``1`` if action ``i``
        contains a tool call, a fenced code block, or a ``\\boxed{}`` answer,
        else ``0``.
    """
    results: List[str] = []
    valids: List[int] = []
    for action in actions:
        text = action if isinstance(action, str) else str(action)
        results.append(text)
        has_signal = bool(
            _RE_TOOL_CALL.search(text)
            or _RE_FENCED_CODE.search(text)
            or _RE_BOXED.search(text)
        )
        valids.append(1 if has_signal else 0)
    return results, valids
