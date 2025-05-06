#! pip install requests
from agno.tools import Toolkit
import requests, io, os
from urllib.parse import urlparse
from typing import IO


class ImageDescriptionTool(Toolkit):
    """
    A simple Agno tool that calls the Qwen‑2‑VL endpoint on Modal
    to obtain a textual description of an image.

    It handles **only**
      - *URL‑based* images, sent as JSON:
        `{ "image": "<url>", "prompt": "<prompt>" }`

    Any other input type (local paths, file objects, Agno `Image` objects, etc.)
    returns an explanatory `[error] …` string instead of raising.
    """

    ENDPOINT = "https://phronetic-ai--qwen-2vl-dapo-multimodal.modal.run"

    def __init__(self) -> None:
        super().__init__(name="ImageDescriptionTool")
        self.register(self.describe_image)

    @staticmethod
    def _is_credentialised(url: str) -> bool:
        low = url.lower()
        return any(tok in low for tok in ("x-amz-", "sig=", "signature="))

    @staticmethod
    def _download_url(url: str) -> IO[bytes]:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
        ext = os.path.splitext(urlparse(url).path)[1] or ".png"
        buf.name = f"download{ext}"
        buf.seek(0)
        return buf

    def describe_image(
        self,
        image_url: str,
        prompt: str = "Describe the contents of the image in detail."
    ) -> str:
        """
        Args:
            image_url (str): **Must** start with ``http://`` or ``https://``.
            prompt (str): Optional extra guidance.

        Returns:
            str: Caption text on success, or ``[error] …`` / ``[exception] …``.
        """
        if not (isinstance(image_url, str) and image_url.startswith(("http://", "https://"))):
            return "[error] Unsupported input type – URL expected."

        try:
            payload = {"prompt": prompt, "image": image_url}
            rsp = requests.post(
                self.ENDPOINT,
                json=payload,
                timeout=120,
                headers={"Accept": "text/plain"},
            )

            if rsp.status_code == 200:
                return rsp.text.strip() or "<empty response>"
            return f"[error] model endpoint returned {rsp.status_code}: {rsp.text[:200]}"

        except Exception as exc:
            return f"[exception] {exc}"
