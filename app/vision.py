import base64
from typing import Optional, Union

import httpx

DEFAULT_CAPTION_PROMPT = (
    "Describe this image factually and concisely, for use as a search-index entry. "
    "If it is a diagram, screenshot, or technical drawing, describe labeled parts, "
    "structure, and any visible text. Do not speculate beyond what is visible."
)


async def describe_image(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    image_bytes: bytes,
    keep_alive: Optional[Union[int, str]] = None,
    prompt: Optional[str] = None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt or DEFAULT_CAPTION_PROMPT,
        "images": [base64.b64encode(image_bytes).decode("ascii")],
        "stream": False,
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    response = await client.post(
        f"{base_url.rstrip('/')}/api/generate",
        json=payload,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()
