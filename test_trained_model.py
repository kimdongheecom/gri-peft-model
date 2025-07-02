#!/usr/bin/env python3
# test_trained_model.py - 훈련된 LoRA 모델 테스트

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

# 경로 설정
BASE_MODEL_PATH = Path(__file__).parent / "models" / "beomi/KoAlpaca-Polyglot-5.8B"
LORA_ADAPTER_PATH = "./lora_results_rtx5060/final_lora_adapter"

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
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 입력 프롬프트 제거하고 응답 부분만 반환
    response = response[len(prompt):].strip()
    
    return response

def main():
    print("=" * 60)
    print("      훈련된 LoRA 모델 테스트")
    print("=" * 60)
    
    # 모델 로드
    model, tokenizer = load_model()
    
    # 테스트 프롬프트 (훈련 데이터와 유사한 형식)
    test_prompt = """You are an expert ESG report writer. Based on the following structured data, synthesize the information into a single, cohesive, and professional Korean paragraph for a sustainability report.

### Data:
{
  "company_info": {
    "name": "테스트기업",
    "id": "company_test"
  },
  "g_standard": "GRI 2: General Disclosures 2021",
  "disclosure_item": "2-1 조직 세부 정보",
  "requirements_and_data": [
    {
      "id": "gri2-1-a",
      "question": "a. 법적 명칭 보고해주세요.",
      "raw_answer": "법적 명칭: 테스트기업 주식회사"
    },
    {
      "id": "gri2-1-b", 
      "question": "b. 소유권 및 법인 구분 보고해주세요.",
      "raw_answer": "주식회사"
    },
    {
      "id": "gri2-1-c",
      "question": "c. 본사 위치 보고해주세요.",
      "raw_answer": "본사 위치: 대한민국 서울특별시 강남구"
    },
    {
      "id": "gri2-1-d",
      "question": "d. 운영 국가(들) 보고해주세요.",
      "raw_answer": "대한민국, 미국"
    }
  ]
}

### Polished Report Paragraph:"""

    print("\n📝 테스트 프롬프트:")
    print("-" * 40)
    print(test_prompt)
    print("-" * 40)
    
    print("\n🤖 모델 응답 생성 중...")
    response = generate_response(model, tokenizer, test_prompt)
    
    print("\n📊 생성된 ESG 보고서:")
    print("=" * 40)
    print(response)
    print("=" * 40)
    
    # 추가 테스트를 위한 인터랙티브 모드
    print("\n💡 추가 테스트를 원하시면 프롬프트를 입력하세요 (종료: 'quit'):")
    
    while True:
        user_input = input("\n프롬프트: ").strip()
        if user_input.lower() in ['quit', 'exit', '종료']:
            break
        
        if user_input:
            print("\n🤖 응답 생성 중...")
            response = generate_response(model, tokenizer, user_input)
            print(f"\n📝 응답:\n{response}")

if __name__ == "__main__":
    main() 