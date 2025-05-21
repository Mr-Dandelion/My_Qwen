import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--image_path", type=str, required=True)
args = parser.parse_args()
model_path = args.model_path
image_path = args.image_path

# Prompt system setup
system_message = (
    "You are VL-Thinking 🤔, a helpful assistant with excellent reasoning ability. "
    "You should first think about the reasoning process and then provide the answer. "
    "Use <think>...</think> and <answer>...</answer> tags."
)

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(model_path)

# Load and preprocess image
if image_path.startswith("http"):
    image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
else:
    image = Image.open(image_path).convert("RGB")
if image.width > 512:
    ratio = image.height / image.width
    image = image.resize((512, int(512 * ratio)), Image.Resampling.LANCZOS)

# Initialize multi-turn chat history
chat = [
    {"role": "system", "content": [{"type": "text", "text": system_message}]},
    {"role": "user", "content": [{"type": "image", "image": image}]}
]

print("💬 多轮对话模式已启动，输入你的问题（输入 'exit' 或 'quit' 结束）：\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 退出对话")
        break

    # 追加用户输入到对话历史
    chat.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

    # 构造输入
    text_input = processor.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text_input], images=[image], return_tensors="pt").to("cuda")

    # 生成回复
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Assistant:\n", output + "\n")

    # 追加模型回复到对话历史
    chat.append({"role": "assistant", "content": [{"type": "text", "text": output}]})

