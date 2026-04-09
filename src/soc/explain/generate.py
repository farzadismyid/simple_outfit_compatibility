from typing import Any

from soc.explain.formatter import build_explanation_payload
from soc.explain.prompt_builder import build_explanation_prompt
from soc.explain.llm_base import BaseLLM


def generate_explanation(
    query_image_path: str,
    target_category: str | None,
    recommendations_df,
    llm: BaseLLM,
    top_k: int = 5,
    query_caption: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end explanation pipeline:
    1. format recommendation results
    2. build grounded prompt
    3. call selected LLM backend
    4. return everything in one dictionary
    """

    payload = build_explanation_payload(
        query_image_path=query_image_path,
        target_category=target_category,
        recommendations_df=recommendations_df,
        top_k=top_k,
        query_caption=query_caption,
    )

    prompt = build_explanation_prompt(payload)
    explanation = llm.generate(prompt)

    return {
        "payload": payload,
        "prompt": prompt,
        "explanation": explanation,
    }
