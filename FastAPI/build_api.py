from typing import List
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import io
from io import BytesIO
import re
import base64
from pydantic import BaseModel
app = FastAPI()

# 模型初始化
base_model_path = "models/Qwen2.5-VL-72B-Instruct"
adapter_path = None

SYSTEM_PROMPT = """你是一个地图专家。需要知道以下信息：
    1.红点为摄像头的点位。
    2.目标只会沿着道路前进。
    图像1是地图信息，图像2是当前摄像头拍摄的目标图片，你需要仔细观察地图和目标图像，
    结合以下几点：
    1.摄像头的朝向。
    2.目标的朝向。
    3.道路信息。
    最后根据用户的要求，智能地寻找摄像头。
    根据问题给出你的推理和思考。
    推理过程和答案分别包含在<think></think>和<answer></answer>标签中，例如：<think>这里是思考过程</think> <answer>这里是回答</answer>。"""

class InferRequest(BaseModel):
    image_base64_list: List[str]
    prompt: str

processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    #quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
model.eval()

# 推理函数
def infer_images(pil_images: List[Image.Image], prompt):
    print("get pil image", flush=True)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": (
                [{"type": "image", "image": img} for img in pil_images] +  # 多图
                [{"type": "text", "text": prompt}]
            )
        }
    ]

    with torch.no_grad():
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        print("get inputs", flush=True)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            max_time=180
        )
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("get response", flush=True)
        print(response, flush=True)
    think_match = re.findall(r"<think>\s*(.*?)\s*</think>", response, re.IGNORECASE | re.DOTALL)
    answer_match = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)

    think_text = think_match[-1].strip() if think_match else ""
    answer_text = answer_match[-1].strip() if answer_match else ""

    return {"think": think_text, "answer": answer_text}


# HTTP接口
@app.post("/infer")
async def infer_api(request: InferRequest):
    try:
        pil_images = [
            Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            for b64 in request.image_base64_list
        ]
        result = infer_images(pil_images, request.prompt)
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
