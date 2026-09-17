from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


def _model_available() -> bool:
    """
    Check whether the local Qwen/PyTorch runtime is installed.

    We deliberately do not download a model from inside an API request.
    """

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False

    return True


def get_model_status() -> Dict[str, Any]:
    """
    Return the current Qwen runtime status.
    """

    available = _model_available()

    return {
        "model": MODEL_NAME,
        "available": available,
        "mode": (
            "local"
            if available
            else "unavailable"
        ),
    }


def _build_context_prompt(
    query: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a grounded semantic-interpretation prompt.

    Qwen is given measured spatial evidence rather than being asked to
    independently invent a temporal change result.
    """

    prompt = f"""
You are the semantic interpretation component of SatQuery AI.

User query:
{query}

Your task is to interpret satellite imagery evidence conservatively.

Important rules:
1. Do not invent observations.
2. Do not invent geographic locations.
3. Do not estimate water depth, water level in metres, or water volume.
4. For water, discuss observable spatial extent/area only.
5. Treat GIS/change-detection measurements as the authoritative source
   for spatial change magnitude and area.
6. If evidence is insufficient, explicitly say so.
7. Do not claim that a change occurred merely because the user asked
   about it.
8. If the chat response needs graphical representation (e.g. data distribution, land cover fractions), output a JSON object representing the data inside a ```pie code block. For example:
```pie
{"Water": 40, "Land": 60}
```
""".strip()

    if evidence:
        prompt += (
            "\n\nMeasured evidence supplied by the spatial analysis:\n"
        )

        for key, value in evidence.items():
            prompt += f"- {key}: {value}\n"

    return prompt


def interpret_change(
    query: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Interpret already-computed spatial evidence.

    This function is intentionally conservative. If the local Qwen runtime
    is unavailable, it returns None so the backend can use its deterministic
    GIS answer rather than fabricating a VLM result.
    """

    if not _model_available():
        return None

    # The actual model loading/inference is intentionally isolated from the
    # core change detector. This keeps the backend usable on machines where
    # PyTorch/Qwen is not installed and prevents model downloads during API
    # requests.
    #
    # The adapted SatQuery-Qwen3-VL-2B runtime can be connected here once the
    # trained checkpoint and its inference environment are available.

    _ = _build_context_prompt(
        query,
        evidence,
    )

    return None


async def run_inference(
    query: str,
    image_paths: List[str],
    tracer: Any = None,
) -> Tuple[Optional[str], List[str]]:
    """
    Qwen semantic-analysis interface.

    At least one image is expected for normal VQA/interpretation.

    The function does not fabricate a response when the local model is
    unavailable or not yet loaded.
    """

    if not image_paths:
        if tracer:
            tracer.append_log(
                "Qwen specialist: no image supplied"
            )

        return (
            None,
            [],
        )

    existing_images = [
        str(Path(path))
        for path in image_paths
        if Path(path).exists()
    ]

    if not existing_images:

        if tracer:
            tracer.append_log(
                "Qwen specialist: supplied image files do not exist"
            )

        return (
            None,
            [],
        )

    status = get_model_status()

    if tracer:
        tracer.append_log(
            "Qwen specialist: "
            f"model={status['model']}, "
            f"available={status['available']}"
        )

    if not status["available"]:

        if tracer:
            tracer.append_log(
                "Qwen specialist unavailable; "
                "no fabricated VLM answer will be returned"
            )

        return (
            None,
            [],
        )

    answer = interpret_change(
        query=query,
        evidence=None,
    )

    if answer is None:

        if tracer:
            tracer.append_log(
                "Qwen runtime detected, but inference checkpoint "
                "is not connected yet"
            )

        return (
            None,
            [],
        )

    return (
        answer,
        existing_images,
    )