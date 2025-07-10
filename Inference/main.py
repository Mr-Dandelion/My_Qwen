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
import csv
import re
import traceback
import glob

SYSTEM_PROMPT = (
    """你是一个地图专家。需要知道以下信息：
    1.红点为摄像头的点位。
    2.目标只会沿着道路前进。
    你需要仔细观察地图，根据路网信息，按照用户的要求智能调取摄像头。
    根据问题给出你的推理和思考。推理过程和答案分别包含在<think></think>和<answer></answer>标签中，例如：<think>这里是思考过程</think> <answer>这里是回答</answer>。"""
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


def find_image_file(base_dir, image_name):
    # 支持的图像扩展名（按优先级排列）
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    for ext in extensions:
        candidate = os.path.join(base_dir, image_name + ext)
        if os.path.isfile(candidate):
            return candidate
    # 如果没有找到，尝试用 glob 模糊查找
    pattern = os.path.join(base_dir, image_name + '.*')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]  # 返回第一个匹配的
    return None

def run_inference_from_csv(base_model_path: str, adapter_path: str,
                           csv_path: str, device: str = "auto"):
    output_rows = []
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
        device_map=device
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    with open(csv_path, newline='', encoding='gbk') as csvfile:
        reader = csv.DictReader(csvfile)  # ✅ 用列名读取
        for idx, row in enumerate(reader):
            if row.get("thinking", "").strip():
                continue

            image_name = row.get("file", "").strip()
            prompt = row.get("question", "").strip()

            image_path = find_image_file("/data/camera_cropped", image_name)
            if not image_path:
                print(f"❌ 图像未找到: {image_name}")
                continue
            try:
                print(f"\n=== 处理第 {idx + 1} 行: 图像={image_path}, 问题={prompt} ===")
                pil_image = load_image(image_path)

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
                    #print(response)
                think_matches = re.findall(r"<think>\s*(.*?)\s*</think>", response, re.IGNORECASE | re.DOTALL)
                answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
                think_text = think_matches[-1].strip() if think_matches else ""
                answer_text = answer_matches[-1].strip() if answer_matches else ""
                print(think_text)
                print(answer_text)
                output_rows.append([image_name, prompt, think_text, answer_text])
            except Exception as e:
                print(f"❌ 处理失败: {image_path}")
                traceback.print_exc()  # ✅ 打印完整错误堆栈
    # 写入输出结果文件
    output_path = "/data/result.csv"
    with open(output_path, "w", newline='', encoding="utf-8-sig") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["image_name", "prompt", "thinking", "answer"])  # 表头
        writer.writerows(output_rows)

    print(f"\n✅ 所有结果已保存至 {output_path}")


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
        help="输入图像的本地文件路径或 URL。"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="需要载入的csv文件"
    )
    parser.add_argument(
        "--prompt",
        type=str,
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

    run_inference_from_csv(
        args.base_model_path,
        args.adapter_path,
        args.csv_path,
        args.device
    )