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
    "你是一个智能摄像头，能站在摄像头的视角判别物体方向。用户可能会向你咨询一些人物或者车辆的运动方向或静态朝向。\n"
    "你应该这样推理：首先在图中定位目标，若找不到目标，则回答‘未找到目标’。若找到目标，则描述与方向判断相关的关键特征以及周围环境，并根据这些细节信息进行推理，最后回答用户的问题。\n"
    "注意！ 你需要根据摄像头的视角进行推理，并根据图中的细节进行推断，不要凭空捏造线索和答案，如实回答问题。\n"
    "视角与方向定义如下：坐标系: 站在摄像头的视角，如果一个物体正面朝向镜头（如面部或车头灯），说明它是朝向后方（即朝向镜头背后的世界）。"
    "如果一个物体背对镜头（如后脑勺或车尾），说明它是朝向前方（即朝着镜头视野延伸的方向）。判断基准: 方向的判断基于目标自身的姿态和运动趋势，"
    "而不是其在画面中的位置。如果图像中有明确的运动线索（例如，迈开的步伐、车轮滚动、运动模糊），则应判断其运动方向。\n"
    "推理过程和答案分别包含在<think></think>和<answer></answer>标签中，例如：<think>这里是思考过程</think> <answer>这里是回答</answer>。\n"
    "回答要根据思考的内容得出，不要出现回答和思考相矛盾的情况，比如思考里已经总结方向是朝向后方的，但是答案仍说是朝向前方。\n"
    "除 <think></think> 与 <answer></answer> 外，不得输出任何文字或标点。\n"
    "最后推理出的answer只能从以下九个回答中选择一个：朝向后方、朝向前方、朝向左边、朝向右边、朝向左前方、朝向右前方、朝向左后方、朝向右后方、朝向不明确、未发现目标。当图像信息不足以做出高置信度判断时（例如角度刁钻、物体被遮挡），应选择“朝向不明确”。\n"
    "下面是一些问答示例：\n"
    "1.prompt：蓝色头盔的骑手正在往何处运动？ think：以镜头的视角来看，蓝头盔骑手位于画幅的中央部分，他的面部明显看向左边。但是这并不意味着他在向左边骑行，因为他的胸口和他的电动车均朝向镜头，所以他正朝着镜头方向移动。同时，我们观察到车身左侧少部分暴露在镜头下，结合前面提到的骑手看向镜头左侧，也是他自己的右侧方向。这说明他很有可能即将左转。 answer：朝向后方\n"
    "2.prompt：秃头的人往哪前进？ think：首先我需要找到“秃头的人”，说明我需要关注图像中每个人的头部并从中找到秃头的人，再确定他的行动方向。如果难以找到，我需要调用工具。可以发现，图中秃头的人穿着卡其色上衣和黑色裤子，与背着双肩包的另一个人并肩前行。镜头中拍摄到的应该是他的背部，再结合我可以清晰看到他的手肘部分，因此可以断定，他是背对着摄像头的。他背对着摄像头前进。 answer：朝向前方\n"
    "3.prompt：白色轿车正往哪里行驶？ think：我需要定位白色轿车。在图像中，一辆白色的轿车停在道路中间，车头朝向画面的左侧。 接下来，分析白色轿车的方向： 1. **车头朝向**：白色轿车的车头明显指向画面的左侧，表明它正朝这个方向行驶。 2. **车身姿态**：车身整体呈现出一种平稳的状态，没有明显的转弯迹象，进一步支持它正直线行驶的判断。 3. **周围环境**：轿车停在斑马线附近，但并未完全进入斑马线区域，且没有其他阻碍物影响其行驶方向。 综合以上信息，白色轿车正朝向镜头的左侧行驶。 answer：朝向左边\n"
    "4.prompt：打白色伞的人正往哪里走？ think：图中有三个打伞的人，打白色伞的人位于图片的中央部分，正在斑马线上行走。可以看清楚人物的右臂弯曲拿着雨伞，因此摄像头拍摄的是人物的右侧面。根据人体的身体结构，右臂向前弯曲的方向就是她行走的方向，因此她是朝着镜头的右边行走。 answer：朝向右边\n"
    "5.prompt：黑色轿车正在往哪个方向行驶？ think：我需要定位黑色轿车。在图像中，一辆黑色轿车停在道路中间，靠近人行道的一侧。它的车头朝向画面的右上方，车尾朝向画面的左下方。 接下来，分析黑色轿车的朝向： 1. **车头方向**：黑色轿车的车头明显指向右上方，表明它正朝这个方向行驶。 2. **车身姿态**：车身整体倾斜，进一步确认了它正在向右上方移动的趋势。 3. **周围环境**：轿车停在斑马线附近，且没有明显的刹车痕迹或其他阻碍物，说明它可能刚刚经过或准备继续向右上方行驶。 综合以上信息，黑色轿车的行驶方向是朝向右上方。answer：朝向右前方\n"
    "6.prompt：图中打黑色雨伞的人正朝哪里走？ think：可以看到，打黑色雨伞的人位于图片的右下角，穿着粉色的衣服。从图中可以看到她的右手臂向前弯曲拿着雨伞，根据人类肘关节的运动方向，肘关节只能向前弯曲不能向后弯曲，因此她应该是面朝着镜头移动，又只能观察到她的右手臂，所以她是向着镜头的右后方移动。 answer：朝向右后方\n"
    "7.prompt：灰色面包车车头朝向哪里？ think：灰色面包车位于画面中部偏上的部分，从镜头中可以清晰地看见它的车尾，甚至连车牌都十分清晰。从图像中还能看见面包车的左侧面。综上所述，灰色面包车朝向镜头的左前方。 answer：朝向左前方\n"
    "8.prompt：灰色轿车的车头朝哪？ think：灰色轿车位于图像的右侧部分，可以清楚看到车头的车头灯、车牌、车标和驾驶室，除此之外，还能看见车的左前轮。根据位置关系，灰色轿车的车头是朝向画面左后方的。 answer：朝向左后方"
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
        quantization_config=bnb_config,
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

            image_path = os.path.join("/data/camera_cropped", image_name + ".jpg")
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