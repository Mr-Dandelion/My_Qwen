from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import io
import re

app = FastAPI()

# 模型初始化
base_model_path = "/home/lanfeng/models/Qwen2.5-VL-72B-Instruct"
adapter_path = None

SYSTEM_PROMPT = "你是一个地图专家。红点为摄像头的点位。目标只会沿着道路前进..."

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map=device
)

model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
model.eval()

# 推理函数
def infer_image(image_bytes, prompt):
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": pil_image},
                                     {"type": "text", "text": prompt}]}
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    think_match = re.findall(r"<think>\s*(.*?)\s*</think>", response, re.IGNORECASE | re.DOTALL)
    answer_match = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)

    think_text = think_match[-1].strip() if think_match else ""
    answer_text = answer_match[-1].strip() if answer_match else ""

    return {"thinking": think_text, "answer": answer_text}


# HTTP接口
@app.post("/infer")
async def infer_api(file: UploadFile, prompt: str = Form(...)):
    try:
        image_bytes = await file.read()
        result = infer_image(image_bytes, prompt)
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
