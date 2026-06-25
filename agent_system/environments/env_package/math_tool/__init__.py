# Copyright 2026 — SOD benchmark (AIME-TIR) integration for SDAR. Apache-2.0.
"""math_tool environment package.

A text environment for tool-integrated reasoning (TIR) on math problems
(the SOD AIME benchmark). Mirrors the `search` env package: the model emits a
Python `code_interpreter` tool call, the env executes it via SandboxFusion and
returns the result, then the model emits a final `\\boxed{...}` answer which is
scored with the strict-box math_dapo scorer.

Exports:
    build_math_tool_envs: factory for the vectorized (threaded) env backend.
    math_tool_projection: maps raw LLM action strings to (results, valids).
"""
from .projection import math_tool_projection
from .envs import build_math_tool_envs
