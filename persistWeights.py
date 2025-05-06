"""
A single time script to mount the weights of Qwen2-VL-DAPO to the modal app - so that the cold start is faster.
"""
import modal
from huggingface_hub import snapshot_download

Volume = modal.Volume.from_name(
    "qwen2vl-cache", create_if_missing=True
)
image = (
    modal.Image.debian_slim()
    .pip_install(
        'huggingface_hub',
        'transformers',
        'tqdm'
    )
)
stub = modal.Stub('qwen-preload')

@stub.function(image=image, volumes={'/root/.cache/huggingface': Volume}, timeout=3000)
def preload():
    snapshot_download(
        "kxxinDave/Qwen-2VL-DAPO",
        local_dir="/root/.cache/huggingface/hub",
        local_files_only=False,
        revision=None
    )