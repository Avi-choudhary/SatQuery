from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

from services import qwen_service


async def run_inference(
    query: str,
    file_paths: List[str],
    tracer: Any,
) -> Tuple[str, List[str]]:
    """
    Single-image VQA specialist.

    Qwen is responsible for semantic interpretation. This service validates
    the input and provides a clean interface for the agent.
    """

    if not file_paths:
        tracer.append_log(
            "VQA specialist: no image supplied"
        )

        return (
            "VQA requires at least one satellite image.",
            [],
        )

    if len(file_paths) > 1:
        tracer.append_log(
            "VQA specialist: multiple images supplied; "
            "using the first image for single-image VQA"
        )

    image_path = Path(
        file_paths[0]
    )

    if not image_path.exists():
        tracer.append_log(
            f"VQA specialist: image not found: {image_path}"
        )

        return (
            "The supplied satellite image could not be found.",
            [],
        )

    tracer.append_log(
        "VQA specialist: validating single-image input"
    )

    answer, evidence = await qwen_service.run_inference(
        query=query,
        image_paths=[str(image_path)],
        tracer=tracer,
    )

    if answer:
        tracer.append_log(
            "VQA specialist: Qwen returned an answer"
        )

        return (
            answer,
            evidence,
        )

    tracer.append_log(
        "VQA specialist: Qwen inference unavailable"
    )

    return (
        "The satellite image was received successfully, "
        "but the Qwen VLM inference runtime/checkpoint is not "
        "currently available on this backend.",
        [str(image_path)],
    )