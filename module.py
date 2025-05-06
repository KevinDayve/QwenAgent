from typing import Optional
from urllib.parse import urlparse
import requests, os, tempfile
from PIL import Image
import io
from agno.tools import Toolkit
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from agno.media import Image as AgnoImage

class QwenModelTool(Toolkit):
    def __init__(self):
        super().__init__(name="qwen_vl_tool")
        if globals()['model'] is None or globals()['processor'] is None: 
            self.model = AutoModelForImageTextToText.from_pretrained(
                "kxxinDave/Qwen-2VL-DAPO",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(
                "kxxinDave/Qwen-2VL-DAPO", use_fast=True
            )
        else:
            self.model = globals()['model']
            self.processor = globals()['processor']
        self.register(self.generate_caption)

    def _loadImage(self, path_or_url: str) -> Image.Image:
        """
        Loads an image from a local file path or URL.
        Arguments:
            path_or_url: str: The local file path or URL to the image.
        Returns:
            Image.Image: The loaded image.
        Raises:
            ValueError: If the image cannot be found or downloaded or if the path is invalid
        """
        parsedUrl = urlparse(path_or_url)
        if parsedUrl.scheme in ("http", "https"):
            try:
                response = requests.get(path_or_url)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                return image
            except requests.exceptions.Timeout:
                raise ValueError(f'Timeout when downloading image from {path_or_url}')
            except requests.exceptions.RequestException as e:
                raise ValueError(f'Error when downloading image from {path_or_url}: {e}')
            except Exception as e:
                raise ValueError(f'Error when downloading image from {path_or_url}: {e}')
        else:
            print(f"Loading image from local path: {path_or_url}")
            fullPath = os.path.abspath(path_or_url)
            if not os.path.exists(fullPath):
                raise ValueError(f'Image not found at {fullPath}')
            if not os.path.isfile(fullPath):
                raise ValueError(f'{fullPath} is not a file')
            try:
                image = Image.open(fullPath).convert("RGB")
                return image
            except Exception as e:
                raise ValueError(f'Error when loading image from {fullPath}: {e}')

    def generate_caption(self, prompt: str, imagePath: str) -> str:
            """
            Generates a textual response (e.g., caption, answer) based on the provided prompt and image.

            This method is automatically exposed as a tool to the host LLM by Agno.
            The LLM will call this tool when it determines it's appropriate based on the user query
            and the tool's description.

            Args:
                prompt (str): The user's textual prompt (e.g., "Describe this image", "What color is the car?").
                            This is extracted by the host LLM from the user's message.
                image (str): The file path or URL of the image to analyze.
                                        Agno should handle passing the correct image reference here,
                                        often derived from an `agno.media.Image` object provided
                                        during the agent call.

            Returns:
                str: The generated text response from the Qwen VL model.

            Raises:
                ValueError: If the model failed to load during initialization or if image loading fails.
                Exception: Can raise exceptions from the underlying model inference.
            """
            print(f"Generating caption for prompt: {prompt} and image: {imagePath}")
            print(f"Type of image: {type(imagePath)}")
            if not self.model or not self.processor:
                raise ValueError("Qwen VL model is not available (failed to load during initialization).")
            image_tensor = self._loadImage(imagePath) if imagePath else None

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_tensor},
                    {"type": "text", "text": prompt},
                ]
            }]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[prompt_text],
                images=[image_tensor] if image_tensor else None,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            out = self.model.generate(**inputs, max_new_tokens=256)
            reply = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            return reply.split("assistant\n", 1)[-1].strip()
