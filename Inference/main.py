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

SYSTEM_PROMPT = (
    # "You are SpacilVLM, a helpful assistant with excellent reasoning ability.\n"
    # "A user asks you a question, and you should try to solve it."
    # "You should first think about the reasoning process in the mind and then provides the user with the answer.\n"
    # "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

    "你是一个能判别物体方向的智能助手。用户可能会向你咨询一些人物或者车辆的运动方向。\n"
    "你应该这样推理：首先在图中定位目标，若找不到目标，则回答‘未找到目标’。若找到目标，"
    "则尽可能详细地描述目标特征以及周围环境，并根据这些细节信息进行推理，最后回答用户的问题。\n"
    "你需要根据图中的细节进行推断，注意不要凭空捏造线索和答案，如实回答问题。\n"
    "如果一个人面部朝向镜头，说明他是面对镜头的后方。如果一个人背对镜头，说明他是面对镜头前方。\n"
    "最后推理出的answer只能从以下八个方向中选择一个：镜头后方（面对镜头）、镜头前方（背对镜头）、镜头左边、镜头右边、镜头左前方、镜头右前方、镜头左后方、镜头右后方。\n"
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
    print(f"正在从 '{base_model_path}' 加载处理器...")
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
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
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

        {"role": "user", "content": [{"type": "image", "image": load_image(few_shot_image_path)},
                                     {"type": "text", "text": "穿黑色衣服的人面朝什么方向？"}, ]},
        {"role": "assistant", "content": [{"type": "text",
                                           "text": "<think>穿黑色衣服的目标较小，可能说明目标离镜头较远。从图像中难以看到人物面部轮廓，说明人物可能是背对镜头。此外穿黑色衣服的人物躯干和裤子均为灰黑的纯色，暗示人物可能背对摄像头。因此，人物背对摄像头。</think>"
                                                   "<answer>背对镜头</answer>"}, ]},
        {"role": "user", "content": [{"type": "image", "image": load_image(few_shot_image_path)},
                                     {"type": "text", "text": "穿米白色衣服的人面朝的是哪个方向？"}, ]},
        {"role": "assistant", "content": [{"type": "text",
                                           "text": "<think>要判断出穿米白色衣服的人面朝的方向，先观察人物面部的朝向，可以较为清晰看到人物的脸，因此人物的脸是面向镜头的。再观察人物的躯干部分。因为人类的手臂不可能以图中这种姿势放在背部，因此人物的手放置在了胸口部分，因此推断出人物躯干的正面也是面向镜头的。综上，人物的面部和躯干都是面向镜头的，所以穿米白色衣服的人是朝向镜头的。</think>"
                                                   "<answer>面向镜头</answer>"}, ]},

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
            if row.get("difficulty", "").strip() != "1":
                continue  # ✅ 只处理难度为1的问题

            image_name = row.get("file", "").strip()
            prompt = row.get("question", "").strip()

            image_path = os.path.join("/data/camera_cropped", image_name + ".jpg")
            try:
                print(f"\n=== 处理第 {idx + 1} 行: 图像={image_path}, 问题={prompt} ===")
                pil_image = load_image(image_path)

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},

                    {"role": "user", "content": [{"type": "image", "image": load_image("/data/camera_cropped/1.jpg")},
                                                 {"type": "text", "text": "穿黑色衣服的人面朝什么方向？"}, ]},
                    {"role": "assistant", "content": [{"type": "text",
                                                       "text": "<think>穿黑色衣服的目标较小，可能说明目标离镜头较远。从图像中难以看到人物面部轮廓，说明人物可能是背对镜头。此外穿黑色衣服的人物躯干和裤子均为灰黑的纯色，暗示人物可能背对摄像头。因此，人物背对摄像头。</think>"
                                                               "<answer>背对镜头</answer>"}, ]},

                    {"role": "user", "content": [{"type": "image", "image": load_image("/data/camera_cropped/1.jpg")},
                                                 {"type": "text", "text": "穿米白色衣服的人面朝的是哪个方向？"}, ]},
                    {"role": "assistant", "content": [{"type": "text",
                                                       "text": "<think>要判断出穿米白色衣服的人面朝的方向，先观察人物面部的朝向，可以较为清晰看到人物的脸，因此人物的脸是面向镜头的。再观察人物的躯干部分。因为人类的手臂不可能以图中这种姿势放在背部，因此人物的手放置在了胸口部分，因此推断出人物躯干的正面也是面向镜头的。综上，人物的面部和躯干都是面向镜头的，所以穿米白色衣服的人是朝向镜头的。</think>"
                                                               "<answer>面对镜头</answer>"}, ]},

                    {"role": "user", "content": [{"type": "image", "image": load_image("/data/camera_cropped/3.jpg")},
                                                 {"type": "text", "text": "蓝色头盔的骑手正在往何处运动？"}, ]},
                    {"role": "assistant", "content": [{"type": "text",
                                                       "text": "<think>以镜头的视角来看，蓝头盔骑手位于画幅的中央部分，他的面部明显看向左边。但是这并不意味着他在向左边骑行，因为他的胸口和他的电动车均朝向镜头，所以他正朝着镜头方向移动。同时，我们观察到车身左侧少部分暴露在镜头下，结合前面提到的骑手看向镜头左侧，也是他自己的右侧方向。这说明他很有可能即将左转。</think>"
                                                               "<answer>面对镜头</answer>"}, ]},

                    {"role": "user", "content": [{"type": "image", "image": load_image("/data/camera_cropped/5.jpg")},
                                                 {"type": "text", "text": "秃头的人往哪前进？"}, ]},
                    {"role": "assistant", "content": [{"type": "text",
                                                       "text": "<think>首先我需要找到“秃头的人”，说明我需要关注图像中每个人的头部并从中找到秃头的人，再确定他的行动方向。如果难以找到，我需要调用工具。可以发现，图中秃头的人穿着卡其色上衣和黑色裤子，与背着双肩包的另一个人并肩前行。镜头中拍摄到的应该是他的背部，再结合我可以清晰看到他的手肘部分，因此可以断定，他是背对着摄像头的。他背对着摄像头前进。</think>"
                                                               "<answer>背对镜头</answer>"}, ]},


                    {"role": "user", "content": [{"type": "image", "image": pil_image},
                                                 {"type": "text", "text": prompt}]}
                ]

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
                #print(response.strip())
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

    # run_inference(
    #     args.base_model_path,
    #     args.adapter_path,
    #     args.image_path,
    #     args.few_shot_image_path,
    #     args.prompt,
    #     args.device
    # )

    run_inference_from_csv(
        args.base_model_path,
        args.adapter_path,
        args.csv_path,
        args.device
    )