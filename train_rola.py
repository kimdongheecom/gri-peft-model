# train.py

import argparse
import torch
import os
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from langchain_community.document_loaders import JSONLoader
import json

# --- 로컬 모델 경로 설정 ---
LOCAL_MODEL_PATH = Path(__file__).parent / "models" / "beomi/KoAlpaca-Polyglot-5.8B"

def main(args):
    print("="*50)
    print("      LoRA Training Script for NVIDIA RTX 5060      ")
    print("      (Using Local Model)                          ")
    print("="*50)

    model_path_to_load = Path(args.model_path)
    if not model_path_to_load.exists() or not (model_path_to_load / "config.json").exists():
        print(f"❌ 오류: 모델을 찾을 수 없습니다. '{model_path_to_load}' 경로를 확인하세요.")
        print("먼저 'python download_model.py' 스크립트를 실행하여 모델을 다운로드하세요.")
        return
    
    print(f"[*] 로컬 모델 경로: {model_path_to_load}")
    print(f"[*] 데이터셋 경로: {args.dataset_path}")
    print(f"[*] 결과 저장 폴더: {args.output_dir}")
    print("-" * 50)

    # 1. 데이터 로딩
    print("[1/4] Loading data...")
    
    # JSONL 파일을 직접 읽어서 처리
    data_list = []
    line_count = 0
    
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # 줄바꿈으로 분할하거나 전체를 하나의 JSON으로 시도
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            line_count += 1
            try:
                json_obj = json.loads(line)
                # prompt와 completion을 결합하여 text 생성
                if 'prompt' in json_obj and 'completion' in json_obj:
                    text = f"{json_obj['prompt']}\n\n{json_obj['completion']}"
                    data_list.append({'text': text})
                    print(f"Successfully loaded record {len(data_list)}")
                else:
                    print(f"Line {i+1}: Missing prompt or completion fields")
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류 at line {i+1}: {e}")
                # 여러 줄에 걸친 JSON일 수 있으므로 전체를 하나로 시도
                try:
                    json_obj = json.loads(content)
                    if 'prompt' in json_obj and 'completion' in json_obj:
                        text = f"{json_obj['prompt']}\n\n{json_obj['completion']}"
                        data_list.append({'text': text})
                        print(f"Successfully loaded single JSON record")
                        break
                except:
                    continue
    
    print(f"Total lines processed: {line_count}")
    print(f"Successfully parsed records: {len(data_list)}")
    
    if len(data_list) == 0:
        print("ERROR: No data loaded. Check the file format.")
        return
    
    dataset = Dataset.from_list(data_list)
    print(f"==> Loaded {len(dataset)} records.")

    # 2. 모델 및 토크나이저 준비
    print("\n[2/4] Preparing model from local path...")
    
    # 4bit quantization 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path_to_load, 
        device_map="auto", 
        trust_remote_code=True, 
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_to_load, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    
    # 4bit 훈련을 위한 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["query_key_value"])
    model = get_peft_model(model, lora_config)
    print("==> Model ready. Trainable parameters:")
    model.print_trainable_parameters()

    # 3. 학습 실행
    print("\n[3/4] Starting training process...")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate=args.learning_rate, 
        num_train_epochs=args.epochs,
        bf16=True, 
        gradient_checkpointing=True,
        logging_steps=10, 
        save_strategy="epoch", 
        max_grad_norm=0.3, 
        lr_scheduler_type="cosine",
        max_seq_length=args.max_seq_len
    )
    trainer = SFTTrainer(
        model=model, 
        train_dataset=dataset, 
        args=sft_config,
        peft_config=lora_config,
        formatting_func=lambda example: example["text"]
    )
    trainer.train()

    # 4. 결과 저장
    print("\n[4/4] Saving the final LoRA adapter...")
    final_adapter_path = os.path.join(args.output_dir, "final_lora_adapter")
    trainer.save_model(final_adapter_path)
    print("="*50)
    print("  TRAINING COMPLETE!  ")
    print(f"==> LoRA adapter saved to: {final_adapter_path}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LoRA Training Script for RTX 5060 from Local Model")
    parser.add_argument('--model_path', type=str, default=str(LOCAL_MODEL_PATH), help="Path to the local base model directory.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to your .jsonl data file.")
    parser.add_argument('--output_dir', type=str, default="./lora_results_rtx5060")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_steps', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    args = parser.parse_args()
    main(args)