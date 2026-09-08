from __future__ import annotations

from typing import Any, Dict, List


class Tracer:
    """
    Lightweight execution tracer for SatQuery.

    Records the major decisions and processing stages so the frontend
    can display an execution summary and the backend can retain evidence
    of how the result was produced.
    """

    def __init__(self) -> None:
        self.state: List[str] = []

    def append_log(
        self,
        step_info: str,
    ) -> None:
        self.state.append(
            str(step_info)
        )

    def get_trace(self) -> Dict[str, Any]:
        return {
            "steps": list(self.state),
            "summary": " | ".join(
                self.state
            ),
        }