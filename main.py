import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--enable_thinking", action='store_true')
parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--do_sample", action='store_true')
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
args = parser.parse_args()
model_path = args.model_path
enable_thinking = args.enable_thinking
max_new_tokens = args.max_new_tokens
do_sample = args.do_sample
temperature = args.temperature
top_p = args.top_p

model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# 初始对话上下文
messages = []

print("🧠 Chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    end = time.time()

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 查找 </think>
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    assistant_reply = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print(f"Qwen: {assistant_reply} ({end - start:.2f}s)\n")

    # 添加助手回复进上下文
    messages.append({"role": "assistant", "content": assistant_reply})
