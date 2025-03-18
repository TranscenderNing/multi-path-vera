import os

os.environ["WANDB_MODE"] = "disabled"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from transformers import Trainer

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, AdaLoraConfig, MELoraConfig
import re
import argparse

model_path = "/data/ldn/llm-models/Meta-Llama-3.1-8B-Instruct"
# lora_path = '/home/ldn/baidu/reft-pytorch-codes/gsm8k-test/lora_model' 



def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer



def predict() -> None:
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.padding_side)
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    
    ds = load_dataset("openai/gsm8k", "main")
    test_ds = ds["test"]
    def modify_question(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example['question']}
        ]
        example['question'] = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        return example

    # 使用 map 应用该函数
    test_ds = test_ds.map(modify_question)
    print(test_ds)
    print(test_ds[0])
    
    batch_size = 2
    batched_test_ds = test_ds.batch(batch_size)

    # 打印查看批次数据
    correct_count = 0
    print("anwser and generation")
    for batch in batched_test_ds:
        model_inputs = tokenizer(batch["question"], return_tensors="pt", padding="longest").to('cuda')
        generated_ids = model.generate(**model_inputs,max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for answer, raw_generation in zip(batch["answer"], response):
            answer = answer.split("####")[-1].strip()
            print(raw_generation)
            generation = extract_answer_number(sentence=raw_generation)
            print(answer, generation)
            if (
                abs(float(extract_answer_number(answer)) - generation)
                <= 0.001
            ):
                correct_count += 1

        break
    print(f"Accuracy: {correct_count / len(test_ds)}")

def train(peft_method=None):

    
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    test_ds = ds["test"]
    # train_ds = train_ds.select(range(5))
    # test_ds = test_ds.select(range(5))
    print(train_ds)
    print(len(train_ds), len(test_ds))
    print(train_ds[0])

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path, use_fast=False, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    def process_func(example):
        MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['answer']}<|eot_id|>", add_special_tokens=False)
        input_ids = (
            instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        )
        attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
        )  # 因为eos token咱们也是要关注的所以 补充为1
        labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
        )
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized_id = train_ds.map(process_func, remove_columns=train_ds.column_names)
    print(tokenized_id)
    print(tokenizer.decode(tokenized_id[0]["input_ids"]))
    print("=" * 100)
    print(
        tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(model)
    print(model.dtype)


    lora_n = 4
    lora_r = [8] * lora_n
    lora_alpha = [16] * lora_n
    print(f"Using melora, lora_r :{lora_r}")
    config = MELoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        mode="melora",
        task_type="CAUSAL_LM",
        init_lora_weights=False
    )
    print("config", config)
    model = get_peft_model(model, config)
    print(model)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=f"./output/llama3_1_instruct_{peft_method}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,  # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    new_model_local = f"{peft_method}_model"
    model.save_pretrained(new_model_local) # Local saving
    tokenizer.save_pretrained(save_directory=new_model_local)
    
    
    # eval 
    tokenizer.padding_side = "left"
    print(tokenizer.padding_side)

    def modify_question(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example['question']}
        ]
        example['question'] = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        return example

    # 使用 map 应用该函数
    test_ds = test_ds.map(modify_question)
    print(test_ds)
    print(test_ds[0])
    
    batch_size = 8
    batched_test_ds = test_ds.batch(batch_size)

    # 打印查看批次数据
    correct_count = 0
    results = []
    print("anwser and generation")
    for batch in batched_test_ds:
        model_inputs = tokenizer(batch["question"], return_tensors="pt", padding="longest").to('cuda')
        generated_ids = model.generate(**model_inputs,max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for question, answer, raw_generation in zip(batch["question"],batch["answer"], response):
            answer = answer.split("####")[-1].strip()
            print(raw_generation)
            generation = extract_answer_number(sentence=raw_generation)
            print(answer, generation)
            if (
                abs(float(extract_answer_number(answer)) - generation)
                <= 0.001
            ):
                correct_count += 1
            results += [
                    {
                        "question": question,
                        "answer": answer,
                        "model_output": generation,
                    }
                ]
                

    print(f"Accuracy: {correct_count / len(test_ds)}")
    with open(f"{peft_method}_results.json", "w") as f:
        json.dump(results, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose PEFT method for training")
    parser.add_argument('--peft_method', type=str, default="lora", 
                        choices=["lora", "dora", "adalora", "pissa", "rslora", "milora","melora"],
                        help="Select PEFT method")

    args = parser.parse_args()
    # lora dora adalora pissa rslora milora
    train(peft_method=args.peft_method)  
    # predict()


if __name__ == "__main__":
    main()
