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
    # "You are SpacilVLM, a helpful assistant with excellent reasoning ability.\n"
    # "A user asks you a question, and you should try to solve it."
    # "You should first think about the reasoning process in the mind and then provides the user with the answer.\n"
    # "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

    "你是 SpacilVLM，一位推理能力超强的得力助手。"
    "用户向您提出问题，您应设法解决。"
    "你应该首先在头脑中思考推理过程，然后向用户提供答案.\n"
    "推理过程和答案分别包含在<think></think>和<answer></answer>标签中，例如：<think>这里是思考过程</think> <answer>这里是回答</answer>."
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


def run_inference(base_model_path: str, adapter_path: str, image_path: str,
                  few_shot_image_path: str, prompt: str, device: str = "auto"):
    print(f"正在从 '{base_model_path}' 加载处理器...")  # ✅ 始终从基础模型目录加载
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
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
        #quantization_config=bnb_config, # 应用于基础模型
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

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},

        # {"role": "user", "content": [{"type": "image", "image": load_image(few_shot_image_path)},
        #                              {"type": "text", "text": "穿黑色衣服的人面朝什么方向？"}, ]},
        # {"role": "assistant", "content": [{"type": "text",
        #                                    "text": "<think>穿黑色衣服的目标较小，可能说明目标离镜头较远。从图像中难以看到人物面部轮廓，说明人物可能是背对镜头。此外穿黑色衣服的人物躯干和裤子均为灰黑的纯色，暗示人物可能背对摄像头。因此，人物背对摄像头。</think>"
        #                                            "<answer>背对镜头</answer>"}, ]},
        # {"role": "user", "content": [{"type": "image", "image": load_image(few_shot_image_path)},
        #                              {"type": "text", "text": "穿米白色衣服的人面朝的是哪个方向？"}, ]},
        # {"role": "assistant", "content": [{"type": "text",
        #                                    "text": "<think>要判断出穿米白色衣服的人面朝的方向，先观察人物面部的朝向，可以较为清晰看到人物的脸，因此人物的脸是面向镜头的。再观察人物的躯干部分。因为人类的手臂不可能以图中这种姿势放在背部，因此人物的手放置在了胸口部分，因此推断出人物躯干的正面也是面向镜头的。综上，人物的面部和躯干都是面向镜头的，所以穿米白色衣服的人是朝向镜头的。</think>"
        #                                            "<answer>朝向镜头</answer>"}, ]},

        {"role": "user", "content": [{"type": "image","image": pil_image}, {"type": "text", "text": prompt},]}
    ]

    text_input_for_processor = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input_for_processor],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )
    target_device = (
        model.device if hasattr(model, "device") and model.device.type != "meta" else
        base_model.device if hasattr(base_model, "device") and base_model.device.type != "meta" else None
    )
    if target_device is not None:
        inputs = inputs.to(target_device)

    print("正在生成回复...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
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
    parser.add_argument(
        "--few_shot_image_path",
        type=str,
        default=None,
        help="输入few-shot图像的本地文件路径或 URL。"
    )
    args = parser.parse_args()

    run_inference(args.base_model_path, args.adapter_path, args.image_path,
                  args.few_shot_image_path, args.prompt, args.device)