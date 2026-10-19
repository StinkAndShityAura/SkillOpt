"""SkillOpt-Sleep — dsh (DeepSeek Harness) session harvesting.

Reads dsh session transcripts (``~/.dsh/sessions/<workspace-slug>/
session-<uuid>/session.v4.jsonl.zstd``) and normalizes them into
:class:`SessionDigest` records without copying tool arguments, private
reasoning blocks, or raw tool outputs.

dsh schema (verified against dsh 0.2.1-alpha.1, session format v4):
  * One zstd-compressed JSONL event stream per session directory. Events
    carry ``seq`` (monotonic) and ``time`` (epoch milliseconds).
  * ``type == "session"`` — exactly one, the first line: ``id``,
    ``createdAt`` (epoch ms), ``cwd``, ``version == 4``.
  * ``type == "user/message"`` — ``data.content`` is a block list. True
    user prompts carry ``data.source.kind == "user"``; rows whose
    ``source.kind`` is ``runtime-context`` / ``time-context`` / anything
    else are host-injected metadata and are skipped — the structural
    equivalent of the legacy ``_is_meta_prompt`` heuristic. ``surfaceOp``
    values other than ``"append"`` mark edits and are skipped too.
  * ``type == "assistant/message"`` — ``data.message.content`` blocks:
    ``"reasoning"`` (private reasoning, skipped), ``"tool-call"`` (only
    the tool ``name`` is kept), ``"text"`` (kept).
  * ``type == "tool/call"`` / ``"tool/result"`` — corroborating tool-name
    source (``data.name``). Result payloads are NOT digested: transient
    per-call errors are normal in agentic coding and are not task
    feedback (same rationale as the pi harvester's ``isError`` note).
  * ``agent/inbox/spliced`` events are inbox edit previews; the landed
    text always re-appears as a ``user/message`` row, so splices are
    ignored to avoid double-counting prompts.

Timestamps are converted from epoch ms to ISO-8601 UTC strings so
``since_iso`` lexicographic comparisons keep working.

This module performs NO writes and NO network calls.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional

from skillopt_sleep.harvest import (
    _detect_feedback,
    _is_headless_replay,
    _is_meta_prompt,
    _project_matches,
    _text_from_content,
)
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import SessionDigest


def _redact_secrets(text: str) -> str:
    """Backward-compatible string wrapper around shared secret redaction."""
    return str(redact_secrets(text))


def _iso_from_ms(value: Any) -> str:
    """Epoch milliseconds -> ISO-8601 UTC string ('' for junk values)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _read_dsh_jsonl(path: str) -> Optional[List[dict[str, Any]]]:
    """Decompress and parse one dsh transcript, rejecting it on corruption.

    A truncated or malformed session is an integrity boundary: returning
    ``None`` skips the whole session instead of exposing an arbitrary
    suffix to the optimizer. Blank lines are tolerated.
    """
    try:
        import zstandard
    except ImportError:
        raise RuntimeError(
            "harvest_dsh requires the 'zstandard' package "
            "(pip install zstandard) to read dsh session files"
        ) from None
    records: List[dict[str, Any]] = []
    try:
        with open(path, "rb") as raw:
            text = zstandard.ZstdDecompressor().stream_reader(raw).read().decode("utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                return None
            records.append(record)
    except (OSError, UnicodeError, ValueError, zstandard.ZstdError):
        return None
    return records


def _tool_names_from_blocks(content: Any) -> List[str]:
    """Tool names from dsh assistant content blocks ({"type": "tool-call"})."""
    names: List[str] = []
    if isinstance(content, list):
        for b in content:
            if (
                isinstance(b, dict)
                and b.get("type") == "tool-call"
                and isinstance(b.get("name"), str)
                and b["name"]
            ):
                names.append(b["name"])
    return names


def _dedup(xs: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def digest_dsh_session(path: str, project: str = "") -> Optional[SessionDigest]:
    """Build a :class:`SessionDigest` from one dsh session transcript."""
    records = _read_dsh_jsonl(path)
    if not records:
        return None
    header = records[0] if records[0].get("type") == "session" else None
    if not isinstance(header, dict) or header.get("version") != 4:
        # Future formats are out of contract; skip rather than guess.
        return None

    session_id = str(header.get("id") or os.path.basename(os.path.dirname(path)))
    session_project = header.get("cwd") if isinstance(header.get("cwd"), str) else ""
    started = _iso_from_ms(header.get("createdAt"))
    ended = started

    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    feedback: List[str] = []
    n_user = 0
    n_asst = 0

    for rec in records[1:]:
        ts = _iso_from_ms(rec.get("time"))
        if ts:
            if not started:
                started = ts
            ended = ts
        data = rec.get("data")
        if not isinstance(data, dict):
            continue
        rtype = rec.get("type")

        if rtype == "user/message":
            source = data.get("source")
            kind = source.get("kind") if isinstance(source, dict) else None
            if kind != "user":
                continue  # runtime-context / time-context / system injections
            if data.get("surfaceOp", "append") != "append":
                continue  # edits of already-counted prompts
            text = _text_from_content(data.get("content"))
            text = _redact_secrets(text).strip()
            if text and not _is_meta_prompt(text):
                n_user += 1
                user_prompts.append(text)
                feedback.extend(_detect_feedback(text))

        elif rtype == "assistant/message":
            message = data.get("message")
            if not isinstance(message, dict):
                continue
            n_asst += 1
            content = message.get("content")
            tools.extend(_tool_names_from_blocks(content))
            text = _text_from_content(content)
            if text.strip():
                assistant_finals.append(_redact_secrets(text).strip())

        elif rtype == "tool/call":
            name = data.get("name")
            if isinstance(name, str) and name:
                tools.append(name)

    if project and not _project_matches(session_project or "", "invoked", project):
        return None
    if n_user == 0 and n_asst == 0:
        return None

    digest = SessionDigest(
        session_id=session_id,
        project=session_project,
        started_at=started,
        ended_at=ended,
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],
        tools_used=_dedup(tools),
        files_touched=[],  # not extractable from dsh transcripts without heuristics
        feedback_signals=_dedup(feedback),
        n_user_turns=n_user,
        n_assistant_turns=n_asst,
        raw_path=path,
    )
    if _is_headless_replay(digest):
        return None
    return digest


def harvest_dsh(
    sessions_dir: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Walk ``~/.dsh/sessions`` (one subdir per workspace slug) and return digests.

    Parameters mirror :func:`skillopt_sleep.harvest.harvest`.
    """
    digests: List[SessionDigest] = []
    if not os.path.isdir(sessions_dir):
        return digests

    paths: List[tuple[float, str]] = []
    for root, _dirs, files in os.walk(sessions_dir):
        for fn in files:
            if fn.endswith(".jsonl.zstd"):
                path = os.path.join(root, fn)
                try:
                    paths.append((os.path.getmtime(path), path))
                except OSError:
                    # A session can disappear while dsh rotates or cleans its
                    # store; skip that file without aborting the whole harvest.
                    continue
    paths.sort(key=lambda item: item[0], reverse=True)

    project_hint = invoked_project if scope == "invoked" else ""
    for _mtime, path in paths:
        digest = digest_dsh_session(path, project=project_hint)
        if digest is None:
            continue
        if not _project_matches(digest.project or "", scope, invoked_project):
            continue
        if since_iso and digest.ended_at and digest.ended_at < since_iso:
            continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break
    return digests
