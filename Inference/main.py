import argparse
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import os
import sys
from peft import PeftModel
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    "You are SpacilVLM, a helpful assistant with excellent reasoning ability.\n"
    "A user asks you a question, and you should try to solve it."
    "You should first think about the reasoning process in the mind and then provides the user with the answer.\n"
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

def load_image(image_path: str) -> Image.Image:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"错误: 图像文件未找到 {image_path}")
        img = Image.open(image_path)
    return img.convert("RGB")


def run_inference(base_model_path: str, adapter_path: str, image_path: str, prompt: str, device: str = "auto"):
    if adapter_path:
        print(f"正在从 '{adapter_path}' 加载处理器...") # 处理器通常和适配器一起保存
        processor = AutoProcessor.from_pretrained(adapter_path,
                                                  trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(base_model_path,
                                                  trust_remote_code=True)

    print("配置 BitsAndBytes (用于4位量化加载)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"步骤 1: 正在从 '{base_model_path}' 加载基础模型...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path, # <--- 明确的基础模型路径
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config, # 应用于基础模型
        low_cpu_mem_usage=True,
        device_map=device
    )

    if adapter_path:
        print(f"步骤 2: 正在从 '{adapter_path}' 加载并应用 LoRA 适配器...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("已加载 LoRA 适配器")
    else:
        model = base_model  # 直接用基础模型
        print("未加载 LoRA，直接使用基础模型")

    print(f"正在加载图像 '{image_path}'...")
    pil_image = load_image(image_path)


    # 构造对话消息，用于视觉预处理
    conversation_for_vision = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image}
            ]
        }
    ]
    # 构造完整对话消息，用于文本模板
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
                #{"type": "image"}
            ]
        }
    ]

    text_input_for_processor = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    vision_img, image_info = process_vision_info(conversation_for_vision)
    inputs = processor(
        text=[text_input_for_processor],
        images=[vision_img],
        return_tensors="pt"
    )
    inputs["image_info"] = image_info
    target_device = (
        model.device if hasattr(model, "device") and model.device.type != "meta" else
        base_model.device if hasattr(base_model, "device") and base_model.device.type != "meta" else None
    )
    if target_device is not None:
        inputs = inputs.to(target_device)

    # ... (生成和解码)
    print("正在生成回复...")
    generated_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )

    print("正在解码回复...")
    response_full = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n模型输出:\n")
    print(response_full.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用微调后的 Qwen2.5VL 模型进行推理。")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="基础模型的 Hugging Face Hub ID 或本地路径。"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA 适配器权重和配置的目录路径。"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="输入图像的本地文件路径或 URL。"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="给模型的文本提示/问题。"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="指定运行设备的字符串 (例如 'cuda', 'cpu', 'auto')。默认为 'auto'。"
    )

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference(args.base_model_path, args.adapter_path, args.image_path, args.prompt, args.device)