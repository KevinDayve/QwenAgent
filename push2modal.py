import modal
from fastapi import HTTPException, UploadFile, Form, File, Request
from typing import Dict, Optional
from urllib.parse import urlparse
from PIL import Image
import requests, io
import torch

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info  # still needed for future tweaks

#Modal app + base image.

app = modal.App("Qwen-2VL-DAPO")

runtimeImage = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "Pillow",
        "accelerate",
        "torch",
        "torchvision",
        "fastapi[standard]",
        "uvicorn",
        "fastapi_mcp",
        "requests",
        "qwen-vl-utils"   # old build tolerates image=None, so you may want to set it to 0.0.14 but we are obviating its use, so it quite doesn't matter here.
    )
)
#TO prevent modal errors (in the future as they are deprecating this in the self-build)
runtimeImage = runtimeImage.add_local_python_source("_remote_module_non_scriptable")
#Lazy Weight loader.

model, processor = None, None #Cached across calls.
#
def loadModel() -> None:
    """Load weights & processor once per container."""
    global model, processor
    if model is None:
        model = AutoModelForImageTextToText.from_pretrained(
            "kxxinDave/Qwen-2VL-DAPO",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            "kxxinDave/Qwen-2VL-DAPO",
            use_fast=True,
        )

#Helper function to fetch image, either from URL or local file.

def fetchImage(imgStr: str) -> Image.Image:
    """Accepts URL or local path, returns RGB PIL.Image"""
    parsed = urlparse(imgStr)
    if parsed.scheme in ("http", "https"):
        resp = requests.get(imgStr, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(imgStr).convert("RGB")


#Please note that in order to keep atleast one container alive (which will keep the model running 24/7 across days of inactivity), set keep_warm, min_container = 1, Currently, to avoid flagrant GPU costs, we set keep_warm=0 and min_containers=0.
volume = modal.Volume.from_name("qwen2vl-cache", create_if_missing=True)
@app.function(image=runtimeImage, gpu="L4", timeout=800, volumes={"/root/.cache/huggingface/hub": volume}, scaledown_window=900, min_containers=0)
@modal.fastapi_endpoint(method="POST")
async def multimodal(
    request: Request,
    prompt: Optional[str] = Form(None),            # for multipart/form‑data
    file:   Optional[UploadFile] = File(None),     # for multipart/form‑data
    image:  Optional[UploadFile] = Form(None),    # form field called image. 
) -> str:
    """
    Accepts **either**:
      • multipart/form‑data → fields:  prompt, file
      • application/json    → keys:    prompt, image  (image = URL or local path)

    Returns the assistant’s plain‑text reply.
    """
    imgPathOrUrl: Optional[str] = None
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        prompt = prompt or body.get("prompt")
        imgPathOrUrl = body.get("image")

    print(f"prompt: {prompt}")
    print(f"imgPathOrUrl: {imgPathOrUrl}")
    if prompt is None and file is None and imgPathOrUrl is None:
        raise HTTPException(
            status_code=422,
            detail="Supply at least one of: prompt, image/file",
        )

    loadModel()
    if file is not None:                              
        bytes_ = await file.read()
        imgTensor = Image.open(io.BytesIO(bytes_)).convert("RGB")
    elif image is not None:
        bytes_ = await image.read()
        imgTensor = Image.open(io.BytesIO(bytes_)).convert("RGB")
    elif imgPathOrUrl:
        imgTensor = fetchImage(imgPathOrUrl)         
    else:
        imgTensor = None

    #Here we run our model.
    contentPayload = []
    if imgTensor is not None:
        contentPayload.append({"type": "image", "image": imgTensor})
    if prompt is not None:
        contentPayload.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": contentPayload}]
    templated = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[templated],
        images=[imgTensor] if imgTensor is not None else None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1024)
    rawReply = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"rawReply: {rawReply}")
    return rawReply.split("assistant\n", 1)[-1].strip()

@app.function(
    image=runtimeImage,
    gpu="L4",
    volumes={"/root/.cache/huggingface/hub": volume},
    min_containers=0,
    scaledown_window=900
)
def warmUp() -> str:
    loadModel()
    return "model ready"

