from typing import Any


def build_explanation_prompt(payload: dict[str, Any]) -> str:
    """
    Build a grounded prompt for an LLM using the structured
    recommendation payload.

    The prompt is intentionally strict so the LLM stays close
    to the provided evidence and avoids making things up.
    """

    query = payload.get("query", {})
    recommendations = payload.get("recommendations", [])

    query_image_path = query.get("image_path", "unknown")
    target_category = query.get("target_category", "unknown")

    lines = []
    lines.append("You are helping explain fashion recommendations to a user.")
    lines.append("Use only the information provided below.")
    lines.append(
        "Do not invent fabrics, occasions, seasons, brands, or fine-grained "
        "visual details unless they are explicitly given."
    )
    lines.append("Keep the explanation short, clear, and user-friendly.")
    lines.append("")
    lines.append("Task:")
    lines.append("Explain why these recommended items may suit the user's outfit.")
    lines.append(
        "Mention general alignment in category, color, style, and overall "
        "visual coherence when supported by the data."
    )
    lines.append("Do not mention technical terms like embeddings, cosine similarity, or CLIP.")
    lines.append("")
    lines.append("Input summary:")
    lines.append(f"- Query image path: {query_image_path}")
    lines.append(f"- Target category: {target_category}")
    lines.append("")

    lines.append("Recommended items:")
    if not recommendations:
        lines.append("- No recommendations were provided.")
    else:
        for item in recommendations:
            rank = item.get("rank")
            item_id = item.get("item_id")
            category = item.get("category")
            score = item.get("score")
            color = item.get("color")
            style = item.get("style")

            item_parts = [
                f"rank={rank}",
                f"item_id={item_id}",
                f"category={category}",
                f"score={score}",
            ]

            if color is not None:
                item_parts.append(f"color={color}")
            if style is not None:
                item_parts.append(f"style={style}")

            lines.append("- " + ", ".join(item_parts))

    lines.append("")
    lines.append("Output requirements:")
    lines.append("- Write one short paragraph, around 3 to 5 sentences.")
    lines.append("- Sound natural and helpful.")
    lines.append("- Focus on why the suggestions are reasonable as complementary items.")
    lines.append("- If the evidence is limited, say so in a subtle natural way without sounding robotic.")

    return "\n".join(lines)
