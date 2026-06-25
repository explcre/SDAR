# Copyright 2026 — SOD benchmark (AIME-TIR) integration for SDAR. Apache-2.0.
"""math_tool environment: tool-integrated reasoning (TIR) for the SOD AIME benchmark.

Mirrors ``agent_system/environments/env_package/search/envs.py`` structure:
    * a per-sample env (``MathToolEnv``) that holds one problem and steps through
      a multi-turn dialogue, and
    * a vectorized, threaded wrapper (``MathToolMultiProcessEnv``) that pads the
      batch to ``env_num * group_n`` with dummy samples + a ``valid_mask`` and
      runs ``reset`` / ``step`` concurrently — exactly like the search wrapper.

The model interacts with the env via the SOD/Qwen *hermes* tool protocol:
    * to call code: ``<tool_call>{"name":"code_interpreter","arguments":{"code":"..."}}</tool_call>``
      The env executes the Python via SandboxFusion and returns
      ``<tool_response>...stdout...</tool_response>`` as the next observation.
    * to answer: emit a final ``\\boxed{...}``. The env scores it with the
      strict-box ``recipe/sod/reward.compute_score`` scorer (the SAME scorer SOD
      uses) and ends the episode.

A ```` ```python ... ``` ```` fenced block is accepted as a fallback code form.

Code execution is delegated to verl's SandboxFusion helper
(``verl.utils.reward_score.sandbox_fusion.utils._process_single_case``); scoring
is delegated to ``recipe/sod/reward.compute_score`` (math_dapo strict box). No
scoring or sandbox logic is reimplemented here.
"""
import asyncio
import concurrent.futures
import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig

try:  # gym is the base class used by every SDAR env wrapper (matches search/envs.py).
    import gym

    _EnvBase = gym.Env
except ImportError:  # keep the CPU test runnable in envs without gym installed.
    class _EnvBase:  # type: ignore
        """Fallback base when ``gym`` is unavailable (CPU-test environments)."""

        pass


# --- action parsing patterns -------------------------------------------------
_RE_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_RE_PY_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_RE_BOXED = re.compile(r"\\boxed\s*{", re.IGNORECASE)


def _default_sandbox_runner(code: str, sandbox_fusion_url: str, timeout: int, language: str) -> str:
    """Run ``code`` in SandboxFusion and return its stdout (or an error string).

    Thin wrapper over verl's ``_process_single_case`` (the same helper
    ``SandboxFusionTool.execute_code`` uses), so the env reuses verl's sandbox
    path instead of reimplementing it. Returns a human-readable string suitable
    to embed in a ``<tool_response>`` observation.
    """
    from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case

    _status, metadata = _process_single_case(
        0, None, None, sandbox_fusion_url, code, timeout, language
    )
    if metadata.get("run_status") == "Finished":
        stdout = metadata.get("stdout")
        stderr = metadata.get("stderr")
        out = stdout if stdout else ""
        if (not out) and stderr:
            out = stderr
        return out if out else ""
    # execution did not finish cleanly (timeout / error)
    err = metadata.get("stderr") or metadata.get("run_status") or "execution failed"
    return f"[code execution error] {err}"


def _extract_code(action: str) -> str | None:
    """Extract Python code from an action: hermes ``<tool_call>`` first, then a fenced block.

    Returns the code string, or ``None`` if no code is present.
    """
    m = _RE_TOOL_CALL.search(action)
    if m:
        raw = m.group(1)
        # strict=False so json accepts LITERAL newlines/tabs inside the "code" string —
        # real models emit multi-line Python with raw \n in the tool_call JSON, which a
        # strict json.loads would reject (then code would never run). (SOD's vLLM hermes
        # parser is likewise tolerant.)
        try:
            payload = json.loads(raw, strict=False)
            args = payload.get("arguments", payload)
            if isinstance(args, str):
                args = json.loads(args, strict=False)
            code = args.get("code")
            if isinstance(code, str):
                return code
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        # last-resort: pull the "code" value directly (handles odd escaping)
        cm = re.search(r'"code"\s*:\s*"(.*)"\s*\}\s*\}?\s*$', raw, re.DOTALL)
        if cm:
            try:
                import codecs
                return codecs.decode(cm.group(1), "unicode_escape")
            except Exception:
                return cm.group(1)
    m = _RE_PY_FENCE.search(action)
    if m:
        return m.group(1).strip()
    return None


def _has_boxed(action: str) -> bool:
    """Return True if the action contains a ``\\boxed{`` final-answer marker."""
    return bool(_RE_BOXED.search(action))


class MathToolEnv:
    """A single AIME tool-integrated-reasoning episode.

    Holds one problem (``question``), its ``ground_truth`` and ``data_source``.
    ``step`` parses one model action and either (a) executes code and returns a
    ``<tool_response>`` observation, or (b) scores a final ``\\boxed{}`` answer
    and ends the episode.

    The sandbox runner is injectable (``sandbox_runner``) so tests can mock it
    without a live sandbox.
    """

    def __init__(self, env_config: DictConfig | None = None, sandbox_runner=None):
        mt_cfg = getattr(env_config, "math_tool", None) if env_config is not None else None
        self.sandbox_fusion_url = (
            getattr(mt_cfg, "sandbox_fusion_url", None) if mt_cfg is not None else None
        ) or "http://localhost:8080/run_code"
        self.timeout = int(getattr(mt_cfg, "timeout", 30)) if mt_cfg is not None else 30
        self.language = getattr(mt_cfg, "language", "python") if mt_cfg is not None else "python"
        self.max_steps = int(getattr(env_config, "max_steps", 16)) if env_config is not None else 16
        # allow tests / callers to override the actual sandbox call
        self._sandbox_runner = sandbox_runner or _default_sandbox_runner

        self.question = ""
        self.ground_truth = ""
        self.data_source = "unknown"
        self.step_count = 0
        self.last_boxed_won = 0.0

    def reset(self, kwargs: Dict[str, Any]) -> Tuple[str, Dict]:
        """Reset the env to a new problem.

        Args:
            kwargs: dict with ``ground_truth``, ``question`` (the full SOD prompt),
                and ``data_source``.

        Returns:
            (obs, info): ``obs`` is the question string; ``info`` carries the
            ``data_source``.
        """
        self.question = kwargs.get("question", "")
        self.ground_truth = str(kwargs.get("ground_truth", ""))
        self.data_source = kwargs.get("data_source", "unknown")
        self.step_count = 0
        self.last_boxed_won = 0.0
        return self.question, {"data_source": self.data_source}

    def _score_boxed(self, action: str) -> float:
        """Score a final boxed answer with the SOD strict-box scorer. Returns won in {0.0, 1.0}."""
        from recipe.sod.reward import compute_score

        res = compute_score(
            data_source=self.data_source,
            solution_str=action,
            ground_truth=self.ground_truth,
        )
        if isinstance(res, dict):
            metric = res.get("acc", res.get("score", 0.0))
        else:
            metric = res
        try:
            return 1.0 if float(metric) > 0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Take one model action.

        Behaviour (in priority order):
            (a) action contains a code tool call -> execute the Python via
                SandboxFusion, return ``<tool_response>...</tool_response>`` as
                the next obs, reward 0.0, done False.
            (b) action contains a final ``\\boxed{}`` -> score it, reward = won,
                done True.
            (c) neither, or the step budget is exhausted -> done True with won
                taken from any boxed answer found (else 0.0).

        Returns:
            (obs, reward, done, info). ``info`` always has ``tool_calling``,
            ``won``, ``is_action_valid`` and ``data_source``.
        """
        self.step_count += 1
        action = action if isinstance(action, str) else str(action)

        code = _extract_code(action)
        # A boxed answer takes precedence ONLY when there is no code call in the
        # same action (a code call means the model is still working).
        if code is not None:
            result = self._sandbox_runner(
                code, self.sandbox_fusion_url, self.timeout, self.language
            )
            obs = f"<tool_response>\n{result}\n</tool_response>"
            done = self.step_count >= self.max_steps
            info = {
                "tool_calling": 1,
                "won": 0.0,
                "is_action_valid": True,
                "data_source": self.data_source,
            }
            if done:
                # ran out of budget mid-tool-use: no answer was produced
                info["won"] = 0.0
            return obs, 0.0, done, info

        if _has_boxed(action):
            won = self._score_boxed(action)
            self.last_boxed_won = won
            info = {
                "tool_calling": 0,
                "won": won,
                "is_action_valid": True,
                "data_source": self.data_source,
            }
            return "", won, True, info

        # (c) neither a tool call nor a boxed answer
        done = self.step_count >= self.max_steps
        won = 0.0
        info = {
            "tool_calling": 0,
            "won": won,
            "is_action_valid": False,
            "data_source": self.data_source,
        }
        obs = "" if done else (
            "<tool_response>\nNo tool call or final answer detected. "
            "Either call the code_interpreter tool or output your final answer in "
            "\\boxed{...}.\n</tool_response>"
        )
        return obs, won, done, info

    def close(self):
        """No-op; the env holds no external resources (sandbox is stateless HTTP)."""
        pass


class MathToolMultiProcessEnv(_EnvBase):
    """Vectorized, threaded wrapper over per-sample ``MathToolEnv``.

    Mirrors ``SearchMultiProcessEnv``: ``total_envs = env_num * group_n``;
    ``reset`` / ``step`` pad the batch to ``total_envs`` with dummy entries +
    a ``valid_mask`` and run concurrently on a thread pool.
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: DictConfig | None = None,
        sandbox_runner=None,
    ) -> None:
        super().__init__()
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.max_steps = env_config.max_steps if env_config is not None else 16
        self._rng = np.random.RandomState(seed)

        self.envs = [
            MathToolEnv(env_config=env_config, sandbox_runner=sandbox_runner)
            for _ in range(self.batch_size)
        ]

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env, kwargs):
        return env.reset(kwargs)

    def _sync_step(self, env, action: str):
        return env.step(action)

    def reset(self, kwargs: List[Dict]):
        """Reset a (possibly short) batch of problems; pads to ``batch_size``.

        Returns ``(obs_list, info_list)`` trimmed back to the real (unpadded) size.
        """
        if len(kwargs) > self.batch_size:
            raise ValueError(
                f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}"
            )

        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {"ground_truth": "", "question": "", "data_source": "unknown"}
        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        return obs_list, info_list

    def step(self, actions: List[str]):
        """Step a (possibly short) batch of actions; pads to ``batch_size``.

        Returns ``(obs_list, reward_list, done_list, info_list)`` trimmed to the
        real (unpadded) size.
        """
        if len(actions) > self.batch_size:
            raise ValueError(
                f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}"
            )

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs_list, reward_list, done_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        return obs_list, reward_list, done_list, info_list

    def close(self):
        """Close all sub-envs and shut down the thread pool / event loop."""
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


def build_math_tool_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None,
    sandbox_runner=None,
):
    """Factory for the vectorized math_tool backend (mirrors ``build_search_envs``).

    Args:
        seed, env_num, group_n, is_train: standard SDAR env factory args.
        env_config: the ``config.env`` DictConfig (reads ``max_steps`` and the
            ``math_tool`` subsection: ``sandbox_fusion_url``, ``timeout``, ``language``).
        sandbox_runner: optional override for the sandbox call (used by CPU tests).
    """
    return MathToolMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
        sandbox_runner=sandbox_runner,
    )
