"""Tunix reward function for Discover computational biology RLVR tasks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

from discover_compbio.tasks import score_completion


logger = logging.getLogger(__name__)

if os.environ.get("COMPBIO_REWARD_PARENT_IGNORE_SIGABRT") == "1":
    try:
        signal.signal(signal.SIGABRT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        logger.warning("Could not install SIGABRT ignore handler for reward parent")


def compbio_reward(prompts, completions, task, answer_json, **kwargs: Any):
    del kwargs

    def as_list(value: Any) -> list[Any]:
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        try:
            return list(value)
        except TypeError:
            return [value]

    prompt_rows = as_list(prompts)
    completion_rows = as_list(completions)
    task_rows = as_list(task)
    answer_rows = as_list(answer_json)
    raise_on_error = os.environ.get("COMPBIO_REWARD_RAISE_ON_ERROR") == "1"
    isolate_score = os.environ.get("COMPBIO_REWARD_ISOLATE_SCORE", "1") != "0"
    subprocess_timeout = float(
        os.environ.get("COMPBIO_REWARD_SUBPROCESS_TIMEOUT_SECONDS", "2400")
    )

    def score_inline(
        task_i: Any, completion: Any, answer_i: Any
    ) -> tuple[float, bool, str | None]:
        try:
            reward = float(score_completion(str(task_i), str(completion), str(answer_i)))
            return reward, True, None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if raise_on_error:
                raise
            logger.exception("compbio_reward failed; using reward=0.0")
            return 0.0, False, f"{type(exc).__name__}: {exc}"

    def score_in_subprocess(
        task_i: Any, completion: Any, answer_i: Any
    ) -> tuple[float, bool, str | None]:
        payload = {
            "task": str(task_i),
            "completion": str(completion),
            "answer_json": str(answer_i),
        }
        child_code = r"""
import json
import sys
import traceback

from discover_compbio.tasks import score_completion

payload = json.load(sys.stdin)
try:
    reward = float(score_completion(
        str(payload["task"]),
        str(payload["completion"]),
        str(payload["answer_json"]),
    ))
except Exception as exc:  # pylint: disable=broad-exception-caught
    print(json.dumps({
        "ok": False,
        "reward": 0.0,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=20),
    }, sort_keys=True))
else:
    print(json.dumps({"ok": True, "reward": reward, "error": None}, sort_keys=True))
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", child_code],
                input=json.dumps(payload),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=subprocess_timeout,
                check=False,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            if raise_on_error:
                raise
            return (
                0.0,
                False,
                f"TimeoutExpired: compbio reward subprocess exceeded "
                f"{subprocess_timeout:.1f}s",
            )
        if proc.returncode != 0:
            if raise_on_error:
                raise RuntimeError(
                    "compbio reward subprocess exited with "
                    f"returncode={proc.returncode}: {proc.stderr[-4000:]}"
                )
            stderr_tail = proc.stderr[-4000:] if proc.stderr else ""
            return (
                0.0,
                False,
                "RewardSubprocessError: "
                f"returncode={proc.returncode}, stderr_tail={stderr_tail!r}",
            )
        try:
            last_line = proc.stdout.strip().splitlines()[-1]
            result = json.loads(last_line)
            reward = float(result.get("reward", 0.0))
            ok = bool(result.get("ok", False))
            error = result.get("error")
            return reward, ok, str(error) if error else None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if raise_on_error:
                raise RuntimeError(
                    "compbio reward subprocess produced invalid output: "
                    f"stdout={proc.stdout[-4000:]!r}, stderr={proc.stderr[-4000:]!r}"
                ) from exc
            return (
                0.0,
                False,
                f"RewardSubprocessProtocolError: {type(exc).__name__}: {exc}",
            )

    rewards: list[float] = []
    audit_rows: list[dict[str, Any]] = []
    for row_idx, completion in enumerate(completion_rows):
        task_i = task_rows[row_idx] if row_idx < len(task_rows) else ""
        answer_i = answer_rows[row_idx] if row_idx < len(answer_rows) else ""
        prompt_i = prompt_rows[row_idx] if row_idx < len(prompt_rows) else ""
        if isolate_score:
            reward, ok, error = score_in_subprocess(task_i, completion, answer_i)
        else:
            reward, ok, error = score_inline(task_i, completion, answer_i)
        rewards.append(reward)
        audit_rows.append(
            {
                "task": str(task_i),
                "prompt": prompt_i,
                "completion": completion,
                "answer_json": str(answer_i),
                "reward": reward,
                "ok": ok,
                "error": error,
            }
        )
    audit_path = os.environ.get("COMPBIO_REWARD_AUDIT_PATH")
    if audit_path:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in audit_rows:
                f.write(json.dumps(row, default=str, sort_keys=True) + "\n")
    return rewards
