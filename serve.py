from fastapi import FastAPI, UploadFile, Form
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import datetime
import io
import requests
from fastapi_mcp import FastApiMCP
import uvicorn
import json

app = FastAPI()
mcp = FastApiMCP(
    app,
    name="Expert Geometry Solver",
    description="A tool that can solve geometry problems.",
    describe_full_response_schema=True,
    describe_all_responses=True
)
mcp.mount()
modelCard = "kxxinDave/Qwen2.5-VL-instruct-3B-Geo"
model = AutoModelForImageTextToText.from_pretrained(
    modelCard,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(modelCard, use_fast=True)

include_operations_mcp = FastApiMCP(
    app,
    name="Geometry Tools",
    include_operations=["generate_image_description"],
)

include_operations_mcp.mount(mount_path="/generate")


@app.post("/generate", operation_id="generate_image_description", include_in_schema=True)
async def generate(
    prompt: str = Form(...),
    image: UploadFile = None,
):
    # Read and process the image if provided
    image_tensor = None
    if image:
        contents = await image.read()
        image_tensor = Image.open(io.BytesIO(contents)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_tensor},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    generated_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    
    image_inputs = [image_tensor] if image_tensor is not None else None
    inputs = processor(
        text=[generated_prompt],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    
    #Run inference.
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    #Log the model responses for future training / analysis.
    log_data = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image.filename if image else None},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": response}
            ]}
        ],
        "tool_name": "GRPO-Qwen2.5-VL-v1",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    with open("log.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")
    response = response.split("assistant\n", 1)[-1].strip()
    return {"response": response}

# For local testing only. Not for deployment...
def test():
    files = {
        'image': None,
        'prompt': (None, 'Describe the image in detail.')
    }
    result = requests.post('http://localhost:8001/generate', files=files)
    print("Status Code:", result.status_code)
    print("Response JSON:", result.json())

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8001)
    test()