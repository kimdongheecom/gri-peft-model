#!/usr/bin/env python3
# test_custom_prompt.py - 특정 프롬프트로 훈련된 LoRA 모델 테스트

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

# 경로 설정
BASE_MODEL_PATH = Path(__file__).parent / "models" / "beomi/KoAlpaca-Polyglot-5.8B"
LORA_ADAPTER_PATH = "./lora_corrected/final_lora_adapter"  # 새로 훈련된 모델

def load_model():
    """훈련된 LoRA 모델을 로드합니다."""
    print("🔄 모델 로딩 중...")
    
    # 4bit quantization 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 베이스 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    print("✅ 모델 로드 완료!")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """주어진 프롬프트에 대해 응답을 생성합니다."""
    # 토크나이즈 (token_type_ids 제외)
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,
        return_token_type_ids=False
    )
    
    # GPU로 이동
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    
    # 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 입력 프롬프트 제거하고 응답 부분만 반환
    response = response[len(prompt):].strip()
    
    return response

def main():
    print("=" * 80)
    print("      새로 훈련된 LoRA 모델 테스트 (116개 샘플)")
    print("=" * 80)
    
    # 모델 로드
    model, tokenizer = load_model()
    
    # 사용자가 제공한 새로운 프롬프트 (GRI 414)
    custom_prompt = """You are an expert ESG report writer. Based on the following structured data, synthesize the information into a single, cohesive, and professional Korean paragraph for a sustainability report.

### Data:
{
  "company_info": {
    "name": "xx기업",
    "id": "company_01"
  },
  "g_standard": "GRI 414: Supplier Social Assessment 2016",
  "disclosure_item": "414-1 사회적 기준에 따른 심사를 거친 신규 공급업체",
  "requirements_and_data": [
    {
      "id": "gri414-1-a",
      "question": "a. 사회적 기준에 따른 심사를 거친 신규 공급업체들의 비중을 보고해주세요.",
      "raw_answer": "모든 신규 공급업체(100%)에 대해 인권, 노동, 안전, 윤리 등 사회적 기준을 포함한 자격 심사를 실시합니다."
    }
  ]
}

### Polished Report Paragraph:"""

    print("\n📝 테스트 프롬프트:")
    print("-" * 60)
    print(custom_prompt)
    print("-" * 60)
    
    print("\n🤖 모델 응답 생성 중...")
    response = generate_response(model, tokenizer, custom_prompt, max_length=800)
    
    print("\n📊 생성된 ESG 보고서:")
    print("=" * 60)
    print(response)
    print("=" * 60)

if __name__ == "__main__":
    main() 