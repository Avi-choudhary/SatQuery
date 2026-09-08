from typing import List

from core.tracer import Tracer
from schemas.responses import SatQueryResponse
from services import change_service, ground_service, vqa_service


# ---------------------------------------------------------------------------
# Query routing vocabulary
# ---------------------------------------------------------------------------

CHANGE_TERMS = (
    "change",
    "changed",
    "changes",
    "between",
    "before",
    "after",
    "temporal",
    "bi-temporal",
    "bitemporal",
    "multitemporal",
    "multi-temporal",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "gain",
    "gained",
    "loss",
    "lost",
    "expanded",
    "expansion",
    "shrunk",
    "reduced",
    "reduction",
    "growth",
    "new building",
    "built-up",
    "built up",
    "water body",
)

GROUNDING_TERMS = (
    "where",
    "location",
    "locate",
    "find",
    "highlight",
    "bbox",
    "bounding box",
    "outline",
    "polygon",
)


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------

def _contains_term(text: str, term: str) -> bool:
    """
    Safe phrase/word matching.

    This avoids accidentally matching pieces of unrelated words.
    """
    import re

    return bool(
        re.search(
            rf"\b{re.escape(term)}\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_change_query(
    query: str,
    file_count: int,
) -> bool:
    """
    Determine whether the request should enter the Change Detective.

    Two uploaded images are treated as the strongest temporal signal because
    the Change Detective requires a before/after pair.

    Cross-modal optical + SAR protection is handled inside change_service.
    """
    if file_count == 2:
        return True

    text = (query or "").lower()

    return any(
        _contains_term(text, term)
        for term in CHANGE_TERMS
    )


def _looks_like_grounding_query(
    query: str,
) -> bool:
    text = (query or "").lower()

    return any(
        _contains_term(text, term)
        for term in GROUNDING_TERMS
    )


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

async def process_query(
    query: str,
    file_paths: List[str],
) -> SatQueryResponse:

    tracer = Tracer()

    tracer.append_log(
        f"Received query: '{query}'"
    )

    tracer.append_log(
        f"Received {len(file_paths)} input file(s)"
    )

    # -----------------------------------------------------------------------
    # Agent routing
    # -----------------------------------------------------------------------

    if _looks_like_change_query(
        query,
        len(file_paths),
    ):
        task = "CHANGE_DETECTION"

    elif _looks_like_grounding_query(query):
        task = "GROUNDING"

    else:
        task = "VQA"

    tracer.append_log(
        f"step 1: classified task as {task}"
    )

    # -----------------------------------------------------------------------
    # Specialist execution
    # -----------------------------------------------------------------------

    try:

        if task == "CHANGE_DETECTION":

            tracer.append_log(
                "step 2: selecting Change Detective specialist"
            )

            text_answer, visual_evidence = (
                await change_service.run_inference(
                    query,
                    file_paths,
                    tracer,
                )
            )

        elif task == "GROUNDING":

            tracer.append_log(
                "step 2: selecting grounding specialist"
            )

            text_answer, visual_evidence = (
                await ground_service.run_inference(
                    query,
                    file_paths,
                    tracer,
                )
            )

        else:

            tracer.append_log(
                "step 2: selecting VQA specialist"
            )

            text_answer, visual_evidence = (
                await vqa_service.run_inference(
                    query,
                    file_paths,
                    tracer,
                )
            )

    except Exception as exc:

        tracer.append_log(
            f"Agent execution error: "
            f"{type(exc).__name__}: {exc}"
        )

        text_answer = (
            "SatQuery could not complete the requested analysis. "
            f"Reason: {exc}"
        )

        visual_evidence = []

    # -----------------------------------------------------------------------
    # Final structured response
    # -----------------------------------------------------------------------

    tracer.append_log(
        "step 3: assembling structured SatQuery response"
    )

    trace_dict = tracer.get_trace()
    return SatQueryResponse(
        text_answer=text_answer,
        visual_evidence=visual_evidence,
        execution_trace=trace_dict,
        trace_log=trace_dict.get("steps", []),
    )