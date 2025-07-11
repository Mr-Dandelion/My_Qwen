import re
import argparse
from dataclasses import dataclass, field
from typing import List
from PIL import Image
from io import BytesIO
import torch
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig, AutoTokenizer)
import os
import sys
import deepspeed

def extract_question(raw_text: str) -> str:
    pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>"
    m = re.search(pattern, raw_text, re.DOTALL)
    return m.group(1).strip() if m else raw_text.strip()


def format_data(sample):
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "你是一个智能摄像头，能站在摄像头的视角判别物体方向。用户可能会向你咨询一些人物或者车辆的运动方向或静态朝向。\n"
                    "你应该这样推理：首先在图中定位目标，若找不到目标，则回答‘未找到目标’。若找到目标，则描述与方向判断相关的关键特征以及周围环境，并根据这些细节信息进行推理，最后回答用户的问题。\n"
                    "注意！ 你需要根据摄像头的视角进行推理，并根据图中的细节进行推断，不要凭空捏造线索和答案，如实回答问题。\n"
                    "视角与方向定义如下：坐标系: 站在摄像头的视角，如果一个物体正面朝向镜头（如面部或车头灯），说明它是朝向后方（即朝向镜头背后的世界）。"
                    "如果一个物体背对镜头（如后脑勺或车尾），说明它是朝向前方（即朝着镜头视野延伸的方向）。判断基准: 方向的判断基于目标自身的姿态和运动趋势，"
                    "而不是其在画面中的位置。如果图像中有明确的运动线索（例如，迈开的步伐、车轮滚动、运动模糊），则应判断其运动方向。\n"
                    "推理过程和答案分别包含在<think></think>和<answer></answer>标签中，例如：<think>这里是思考过程</think> <answer>这里是回答</answer>。\n"
                    "回答要根据思考的内容得出，不要出现回答和思考相矛盾的情况，比如思考里已经总结方向是朝向后方的，但是答案仍说是朝向前方。\n"
                    "除 <think></think> 与 <answer></answer> 里面的内容外，不得输出任何文字或标点。\n"
                )
            }
        ]
    }
    formatted = [system_message]

    user_msg = {"role": "user", "content": []}
    question = extract_question(sample.get("input", ""))
    if question:
        user_msg["content"].append({"type": "text", "text": question})
    images = sample.get("images") or []
    if images:
        user_msg["content"].append({"type": "image", "image": images})
    formatted.append(user_msg)

    if sample.get("output"):
        formatted.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["output"]}]
        })
    return formatted


def collate_fn(examples, processor):
    actual_conversations_raw = [sample['messages'] for sample in examples]

    cleaned_conversations = []
    for conversation in actual_conversations_raw:
        new_cleaned_conversation_messages = []
        for message_dict in conversation:
            original_content_parts = message_dict.get('content', [])
            new_message_content_parts = []

            for part in original_content_parts:
                current_part = part.copy()

                if current_part.get('type') == 'text':
                    if current_part.get('text') is None:
                        continue
                    if 'image' in current_part and current_part.get('image') is None:
                        del current_part['image']
                    new_message_content_parts.append(current_part)
                elif current_part.get('type') == 'image':
                    image_data = current_part.get('image')
                    if image_data is None:
                        continue
                    if isinstance(image_data, dict) and 'bytes' in image_data:
                        try:
                            pil_image = Image.open(BytesIO(image_data['bytes']))
                            current_part['image'] = pil_image  # 将字典替换为PIL对象
                        except Exception as e:
                            print(f"警告: 无法将字节转换为PIL图像: {e}。跳过此图像。")
                            continue
                    elif not isinstance(image_data, Image.Image):  # 如果不是字典也不是PIL Image，则可能是qwen_vl_utils无法处理的类型
                        print(f"警告: 未知的图像数据类型 {type(image_data)}。跳过此图像。")
                        continue
                    if 'text' in current_part and current_part.get('text') is None:
                        del current_part['text']
                    new_message_content_parts.append(current_part)
                else:
                    new_message_content_parts.append(current_part)

            if new_message_content_parts:
                new_message = message_dict.copy()
                new_message['content'] = new_message_content_parts
                new_cleaned_conversation_messages.append(new_message)

        if new_cleaned_conversation_messages:
            cleaned_conversations.append(new_cleaned_conversation_messages)

    texts = []
    image_batches = []
    if not cleaned_conversations:
        print("警告: 清理后，批次中的所有对话都变为空。")
    else:
        texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in cleaned_conversations]
        image_batches = [process_vision_info(conversation)[0] for conversation in cleaned_conversations]

    if not texts and not any(img is not None for img in image_batches):
        if not cleaned_conversations:
             print("错误: 清理后没有有效的对话来处理。")
             return {}

    batch = processor(text=texts, images=image_batches, return_tensors="pt", padding=True)
    max_len = processor.tokenizer.model_max_length
    for idx, input_ids in enumerate(batch["input_ids"]):
        if input_ids.shape[0] > max_len:
            print(f"[警告] 第 {idx} 个样本 tokens 长度 {input_ids.shape[0]} 超过最大长度 {max_len}")
    batch = {k: v.cpu() for k, v in batch.items()}

    if not batch:
        return batch

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_token_ids = (
        [151652, 151653, 151655]
        if hasattr(processor, "image_processor")
        else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    )
    for tid in image_token_ids:
        labels[labels == tid] = -100

    batch["labels"] = labels
    return batch



@dataclass
class TrainingConfig:
    model_id: str = "/home/lanfeng/models/Qwen2.5-VL-7B-Instruct"
    dataset_id: str = "/home/lanfeng/Datasets/My_SFT"
    image_root: str = "camera_cropped"
    messages_file: str = "messages.csv"
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.5
    target_modules: List[str] = field(default_factory=lambda:
    ["q_proj", "v_proj", "k_proj", "o_proj", "qkv", "proj"])
    num_train_epochs: int = 1
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    output_dir: str = "/home/lanfeng/Checkpoints/Qwen2.5VL-7B-lora"
    deepspeed_config: str = field(default=None,
                                  metadata={"help": "Path to the DeepSpeed config file (e.g., ds_config.json)."})
    local_rank: int = field(default=-1)


def parse_args() -> TrainingConfig:
    default_cfg = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train a VL Spacethinker model with LoRA")
    parser.add_argument("--model_id", default=default_cfg.model_id)
    parser.add_argument("--dataset_id", default=default_cfg.dataset_id)
    parser.add_argument("--image_root", default=default_cfg.image_root )
    parser.add_argument("--messages_file", default=default_cfg.messages_file)
    parser.add_argument("--lora_r", type=int, default=default_cfg.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=default_cfg.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=default_cfg.lora_dropout)
    parser.add_argument(
        "--target_modules",
        default=','.join(default_cfg.target_modules),
        help="Comma-separated list of target modules for LoRA"
    )
    parser.add_argument("--num_train_epochs", type=int, default=default_cfg.num_train_epochs)
    parser.add_argument("--train_batch_size", type=int, default=default_cfg.train_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=default_cfg.eval_batch_size)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=default_cfg.gradient_accumulation_steps
    )
    parser.add_argument("--learning_rate", type=float, default=default_cfg.learning_rate)
    parser.add_argument("--output_dir", default=default_cfg.output_dir)
    parser.add_argument("--deepspeed_config", type=str, default=default_cfg.deepspeed_config,
                        help="Path to the DeepSpeed config file.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training, passed by DeepSpeed launcher")
    args = parser.parse_args()
    return TrainingConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        image_root=args.image_root,
        messages_file=args.messages_file,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        deepspeed_config=args.deepspeed_config,
        local_rank=args.local_rank,
    )

def build_output(row) -> str:
    thinking = str(row.get("thinking", "")).strip()
    answer   = str(row.get("answer", "")).strip()
    has_thinking = (thinking != "") and (not (thinking.lower() in {"nan", "none"}))
    if has_thinking:
        return f"<think>{thinking}</think>\n<answer>{answer}</answer>"
    else:
        return f"<answer>{answer}</answer>"



def prepare_datasets(cfg: TrainingConfig):
    try:
        df = pd.read_csv(os.path.join(cfg.dataset_id, cfg.messages_file), keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        print("[INFO] 不是 UTF-8，改用 GBK 打开 …")
        df = pd.read_csv(os.path.join(cfg.dataset_id, cfg.messages_file), keep_default_na=False, encoding="gbk")

    df = df.rename(columns={"question": "input"})
    df["output"] = df.apply(build_output, axis=1)
    def to_image(item):
        path = os.path.join(cfg.dataset_id, cfg.image_root, f"{item}.jpg")  # 按需改扩展名
        return Image.open(path).convert("RGB") if os.path.exists(path) else None
    df["images"] = df["file"].apply(to_image)
    records = df[["input", "output", "images"]].to_dict(orient="records")
    split_idx = int(len(records) * 0.9)
    train_ds = Dataset.from_list([
        {"messages": format_data(r), "input_ids": []}
        for r in tqdm(records[:split_idx], desc="Train")
    ])
    eval_ds   = Dataset.from_list([
        {"messages": format_data(r), "input_ids": []}
        for r in tqdm(records[split_idx:], desc="Eval")
    ])
    return train_ds, eval_ds


def prepare_model_and_optimizer(cfg: TrainingConfig):
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        trust_remote_code=True
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    device_to_load_on = cfg.local_rank
    if cfg.local_rank == -1:
        if torch.cuda.is_available():
            device_to_load_on = "cuda"
        else:
            device_to_load_on = "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb,
        low_cpu_mem_usage=True,
        device_map=device_to_load_on
    )
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, processor, peft_cfg



def main():
    cfg = parse_args()
    raw_train_ds, raw_eval_ds = prepare_datasets(cfg)
    print('finished loading datasets')
    model, processor, peft_cfg = prepare_model_and_optimizer(cfg)
    print('finished model')
    sft_training_args_dict = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.train_batch_size,
        "per_device_eval_batch_size": cfg.eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 50,
        "max_steps": 30,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "fp16": True,
        "max_grad_norm": 0.3,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_dir": f"{cfg.output_dir}/logs",
        "save_total_limit": 2,
        "remove_unused_columns": False,
    }

    if cfg.deepspeed_config:
        sft_training_args_dict["deepspeed"] = cfg.deepspeed_config
        print(f"Using DeepSpeed with config: {cfg.deepspeed_config}")
    else:
        print("Not using DeepSpeed (no config file provided).")

    training_args = SFTConfig(**sft_training_args_dict)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_train_ds,  # 使用原始的训练数据集
        eval_dataset=raw_eval_ds,
        peft_config=peft_cfg,
        data_collator=lambda data: collate_fn(data, processor),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final PEFT adapter and processor to {cfg.output_dir}")
    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)

    print("Training finished.")


if __name__ == "__main__":
    main()