#!/usr/bin/env python
# Copyright 2026 — SOD benchmark (AIME-TIR) integration for SDAR. Apache-2.0.
"""CPU-only smoke test for the math_tool (AIME-TIR) environment backend.

Verifies WITHOUT a model, GPU, or live sandbox:
  (a) reset(kwargs) returns the question;
  (b) a final \\boxed{33} answer with ground_truth '33' -> done=True, won=1.0;
      a wrong boxed answer -> won=0.0;
  (c) a code tool call -> done=False and a <tool_response> observation
      (the sandbox call is MOCKED via an injected sandbox_runner);
  (d) projection validity flags.

Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/pengchx3/text-dna/SDAR-sod \\
      /home/pengchx3/miniconda/envs/sdar/bin/python recipe/sod/test_math_tool_env.py
"""
import sys

from agent_system.environments.env_package.math_tool import (
    build_math_tool_envs,
    math_tool_projection,
)
from agent_system.environments.env_package.math_tool.envs import MathToolEnv


def _fake_sandbox(code, sandbox_fusion_url, timeout, language):
    """Mock sandbox: pretend the code printed '42' (no live SandboxFusion needed)."""
    return "42\n"


def test_reset_returns_question():
    """(a) reset(kwargs) returns the stored question string."""
    env = MathToolEnv(sandbox_runner=_fake_sandbox)
    obs, info = env.reset({"question": "What is 6*7?", "ground_truth": "42", "data_source": "aime2024"})
    assert obs == "What is 6*7?", obs
    assert info["data_source"] == "aime2024", info
    print("PASS (a) reset returns question")


def test_boxed_scoring():
    """(b) correct boxed -> won=1.0/done; wrong boxed -> won=0.0/done."""
    env = MathToolEnv(sandbox_runner=_fake_sandbox)
    env.reset({"question": "q", "ground_truth": "33", "data_source": "aime2024"})
    obs, reward, done, info = env.step("After reasoning, the final answer is \\boxed{33}.")
    assert done is True, done
    assert info["won"] == 1.0, info
    assert reward == 1.0, reward
    assert info["tool_calling"] == 0, info

    env2 = MathToolEnv(sandbox_runner=_fake_sandbox)
    env2.reset({"question": "q", "ground_truth": "33", "data_source": "aime2024"})
    obs2, reward2, done2, info2 = env2.step("The final answer is \\boxed{99}.")
    assert done2 is True, done2
    assert info2["won"] == 0.0, info2
    assert reward2 == 0.0, reward2
    print("PASS (b) boxed scoring: correct=1.0, wrong=0.0")


def test_code_tool_call():
    """(c) a hermes code tool call -> done=False + <tool_response> obs (sandbox mocked)."""
    env = MathToolEnv(sandbox_runner=_fake_sandbox)
    env.reset({"question": "q", "ground_truth": "42", "data_source": "aime2024"})
    action = '<tool_call>{"name": "code_interpreter", "arguments": {"code": "print(6*7)"}}</tool_call>'
    obs, reward, done, info = env.step(action)
    assert done is False, done
    assert "<tool_response>" in obs and "42" in obs, obs
    assert info["tool_calling"] == 1, info
    assert reward == 0.0, reward

    # fenced-block fallback also routes to the sandbox
    env3 = MathToolEnv(sandbox_runner=_fake_sandbox)
    env3.reset({"question": "q", "ground_truth": "42", "data_source": "aime2024"})
    obs3, _, done3, info3 = env3.step("Let me compute:\n```python\nprint(6*7)\n```")
    assert done3 is False and info3["tool_calling"] == 1, (done3, info3)
    assert "<tool_response>" in obs3 and "42" in obs3, obs3
    print("PASS (c) code tool call -> tool_response, done=False (sandbox mocked)")


def test_projection_validity():
    """(d) projection: tool call / fenced code / boxed are valid; plain text invalid."""
    actions = [
        '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(1)"}}</tool_call>',
        "```python\nprint(1)\n```",
        "answer is \\boxed{7}",
        "just some reasoning with no action",
    ]
    results, valids = math_tool_projection(actions)
    assert results == actions, "projection should pass action text through"
    assert valids == [1, 1, 1, 0], valids
    print("PASS (d) projection validity flags")


def test_vectorized_backend_pad_and_mask():
    """Bonus: vectorized backend reset/step pad to batch_size and trim back."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"max_steps": 4, "math_tool": {"sandbox_fusion_url": "http://x/run_code"}})
    backend = build_math_tool_envs(
        seed=0, env_num=2, group_n=1, is_train=False, env_config=cfg, sandbox_runner=_fake_sandbox
    )
    # only 1 real sample for a batch_size of 2 -> exercises padding + valid_mask
    kwargs = [{"question": "q1", "ground_truth": "5", "data_source": "aime2025"}]
    obs, infos = backend.reset(kwargs)
    assert obs == ["q1"], obs
    assert len(infos) == 1, infos
    obs2, rewards, dones, infos2 = backend.step(["\\boxed{5}"])
    assert len(obs2) == 1 and rewards[0] == 1.0 and dones[0] is True, (obs2, rewards, dones)
    backend.close()
    print("PASS (e) vectorized backend padding + valid_mask")


if __name__ == "__main__":
    test_reset_returns_question()
    test_boxed_scoring()
    test_code_tool_call()
    test_projection_validity()
    test_vectorized_backend_pad_and_mask()
    print("\nALL MATH_TOOL ENV CPU TESTS PASSED")
    sys.exit(0)
