"""Findings ledger sidecar for MedRAX (SPEC-LEDGER-001).

Gives every tool call structured, auditable provenance without touching the
``process <-> execute`` graph. Two pieces:

* ``build_envelope`` — a JSON-safe envelope (tool, inputs, raw output, source,
  timestamp, optional confidence) attached additively to each ``ToolMessage``.
* ``FindingsLedger`` — an external store keyed by ``thread_id``, file-backed
  under the agent's log directory (retention parity with the existing logs).

The envelope is supplementary metadata; ``ToolMessage.content`` is left untouched
so existing string consumers (e.g. ``interface.py`` ``eval(message.content)[0]``)
keep working.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Orchestration tool name -> upstream evidence-source / model identity (REQ-1).
# This is the model that actually produced the finding, kept distinct from the
# tool name so ``provenance_source`` is meaningful rather than a restatement.
TOOL_SOURCE_MODEL: Dict[str, str] = {
    "chest_xray_classifier": "DenseNet-121",
    "chest_xray_segmentation": "MedSAM",
    "chest_xray_grounding": "Maira-2",
    "chest_xray_report_generator": "SwinV2-CheXpert",
    "chest_xray_vqa": "CheXagent",
    "llava_med": "LLaVA-Med",
    "image_generator": "RoentGen",
}


def resolve_provenance_source(tool_name: str) -> str:
    """Resolve the upstream evidence-source identity for a tool (REQ-1).

    Falls back to ``tool:{tool_name}`` when no distinct upstream model is mapped.
    """
    return TOOL_SOURCE_MODEL.get(tool_name, f"tool:{tool_name}")


def _to_json_safe(value: Any) -> Tuple[Any, bool]:
    """Return ``(json_safe_value, had_unserializable)``.

    Tuples are normalized to lists and containers recurse. A value json cannot
    encode is string-coerced and ``had_unserializable`` is set True. Tuple->list
    normalization alone does NOT set the flag — only genuine coercion does.
    """
    if isinstance(value, (list, tuple)):
        flagged = False
        out: List[Any] = []
        for item in value:
            safe, item_flag = _to_json_safe(item)
            out.append(safe)
            flagged = flagged or item_flag
        return out, flagged
    if isinstance(value, dict):
        flagged = False
        out_map: Dict[str, Any] = {}
        for key, item in value.items():
            safe, item_flag = _to_json_safe(item)
            out_map[str(key)] = safe
            flagged = flagged or item_flag
        return out_map, flagged
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, False
    return str(value), True


def extract_confidence(raw_output: Any) -> Optional[float]:
    """Best-effort confidence proxy (REQ-5).

    If the tool output is a ``(payload, metadata)`` pair (or a bare dict) whose
    payload is a dict of numeric scores in ``[0, 1]``, return the max. Returns
    ``None`` when no such signal exists — calibration is intentionally out of
    scope (the field is then omitted from the envelope).
    """
    payload = raw_output
    if isinstance(raw_output, (list, tuple)) and raw_output:
        payload = raw_output[0]
    if isinstance(payload, dict) and payload:
        values = list(payload.values())
        if all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1
            for v in values
        ):
            return max(values)
    return None


def build_envelope(
    tool_name: str,
    tool_call_id: str,
    args: Any,
    raw_output: Any,
    *,
    raised: bool = False,
    confidence: Optional[float] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the JSON-safe provenance envelope for one tool call (REQ-1, REQ-8b).

    ``error`` is True when the tool raised (``raised``) or returned a value that
    had to be string-coerced.
    """
    safe_args, _ = _to_json_safe(args)
    safe_output, had_unserializable = _to_json_safe(raw_output)
    envelope: Dict[str, Any] = {
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "args": safe_args,
        "raw_output": safe_output,
        "provenance_source": resolve_provenance_source(tool_name),
        "timestamp": (now or datetime.now()).isoformat(),
        "error": bool(raised or had_unserializable),
    }
    if confidence is not None:
        envelope["confidence"] = confidence
    return envelope


class FindingsLedger:
    """External, ``thread_id``-keyed store of finding envelopes (REQ-3, REQ-10).

    File-backed under the agent's log directory for retention parity with the
    existing per-batch tool-call logs. No separate rotation/retention surface is
    introduced by this feature.
    """

    def __init__(self, log_dir: str = "logs", filename: str = "ledger.json") -> None:
        self.path = Path(log_dir) / filename
        self._index: Dict[str, List[Dict[str, Any]]] = {}

    def append(self, thread_id: Optional[str], entry: Dict[str, Any]) -> None:
        key = thread_id or "unknown"
        self._index.setdefault(key, []).append(entry)
        self._flush()

    def get(self, thread_id: Optional[str]) -> List[Dict[str, Any]]:
        return list(self._index.get(thread_id or "unknown", []))

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self._index, handle, indent=2)
