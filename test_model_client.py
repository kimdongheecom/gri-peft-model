# test_model_client.py
# KoAlpaca 5.8B 모델 테스트 클라이언트 (RTX 5060 최적화)

import os
import logging
import traceback
import time
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# RTX 5060 (Ada Lovelace) 호환성 설정
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 설정 ---
MODEL_ID = "beomi/KoAlpaca-Polyglot-5.8B"
MODEL_PATH = Path(__file__).parent / "models" / MODEL_ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class KoAlpacaModelTester:
    """KoAlpaca 5.8B 모델 직접 테스트 클래스"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = DEVICE
        logger.info(f"ModelTester 초기화 (Device: {self.device})")
        
    def check_model_exists(self) -> bool:
        """모델이 로컬에 존재하는지 확인"""
        config_file = MODEL_PATH / "config.json"
        model_files = list(MODEL_PATH.glob("*.safetensors")) if MODEL_PATH.exists() else []
        
        exists = config_file.exists() and len(model_files) > 0
        if exists:
            logger.info(f"✅ 모델 발견: {MODEL_PATH}")
            logger.info(f"   - 설정 파일: {config_file}")
            logger.info(f"   - 모델 파일 수: {len(model_files)}")
        else:
            logger.error(f"❌ 모델을 찾을 수 없습니다: {MODEL_PATH}")
            logger.error("💡 'python download_model.py'를 먼저 실행해주세요.")
        
        return exists
    
    def load_model(self):
        """모델을 메모리에 로드"""
        if not self.check_model_exists():
            raise FileNotFoundError("모델이 존재하지 않습니다. download_model.py를 먼저 실행해주세요.")
        
        try:
            logger.info("🧠 KoAlpaca 5.8B 모델 로딩 시작...")
            
            # RTX 5060 호환성을 위한 설정
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="eager",  # Flash Attention 비활성화
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ 모델 및 토크나이저 로딩 완료!")
            
            # GPU 메모리 사용량 확인
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🔥 GPU 메모리 사용량: {memory_allocated:.2f} GB")
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            traceback.print_exc()
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """프롬프트에 대한 응답 생성"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        # KoAlpaca 대화형 프롬프트 템플릿
        conversation_prompt = f"""### 질문: {prompt}

### 답변:"""
        
        logger.info(f"🤖 응답 생성 중... (최대 {max_new_tokens} 토큰)")
        
        try:
            inputs = self.tokenizer(conversation_prompt, return_tensors="pt", add_special_tokens=True)
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            inputs = inputs.to(self.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                except RuntimeError as e:
                    if "no kernel image is available" in str(e):
                        logger.warning("CUDA 커널 호환성 문제 감지. 안전 모드로 재시도...")
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=min(max_new_tokens, 50),
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=False,
                        )
                    else:
                        raise
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 답변 부분만 추출
            answer_start = response_text.find("### 답변:") + len("### 답변:")
            answer = response_text[answer_start:].strip()
            
            # 다음 질문이 시작되면 그 전까지만 반환
            if "### 질문:" in answer:
                answer = answer.split("### 질문:")[0].strip()
            
            logger.info(f"⏱️ 생성 시간: {generation_time:.2f}초")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ 응답 생성 실패: {e}")
            raise

def test_model_connection(tester: KoAlpacaModelTester):
    """모델 연결 및 로딩 테스트"""
    try:
        logger.info("🔧 모델 연결 테스트 시작...")
        tester.load_model()
        logger.info("✅ 모델 연결 성공!")
        return True
    except Exception as e:
        logger.error(f"❌ 모델 연결 실패: {e}")
        return False

def run_interactive_chat(tester: KoAlpacaModelTester):
    """대화형 챗봇 모드"""
    print("\n🤖 KoAlpaca 5.8B 대화형 챗봇 시작!")
    print("💡 'quit', 'exit', '종료'를 입력하면 종료됩니다.")
    print("💡 'clear'를 입력하면 GPU 메모리를 정리합니다.")
    print("="*60)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("👋 챗봇을 종료합니다.")
                break
            if user_input.lower() == 'clear':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("🧹 GPU 메모리 캐시 정리 완료")
                continue
            if not user_input:
                continue

            print("🤖 Bot: (응답 생성 중...)", end="\r", flush=True)
            
            answer = tester.generate_response(user_input)
            
            # 응답 출력
            print("🤖 Bot: " + " " * 20)  # "응답 생성 중..." 메시지 덮어쓰기
            print(answer)
            
            # GPU 메모리 사용량 표시
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"🔥 GPU 메모리: {memory_used:.2f} GB")

        except KeyboardInterrupt:
            print("\n\n👋 사용자가 중단했습니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            logger.error(f"대화 처리 오류: {e}")

def run_benchmark(tester: KoAlpacaModelTester):
    """모델 성능 벤치마크"""
    print("\n📊 KoAlpaca 5.8B 성능 벤치마크 시작!")
    print("="*60)

    test_cases = [
        ("인사", "안녕하세요! 자기소개를 해주세요."),
        ("일반 지식", "대한민국의 수도와 인구를 알려주세요."),
        ("설명 요청", "인공지능이란 무엇인지 간단히 설명해주세요."),
        ("창작 요청", "봄에 대한 짧은 시를 써주세요."),
        ("문제 해결", "파이썬에서 리스트를 정렬하는 방법을 알려주세요."),
    ]
    
    total_time = 0
    total_tokens = 0
    successful_tests = 0

    for i, (test_name, prompt) in enumerate(test_cases, 1):
        print(f"\n🧪 테스트 {i}/{len(test_cases)}: {test_name}")
        print(f"   프롬프트: {prompt}")
        
        try:
            start_time = time.time()
            answer = tester.generate_response(prompt, max_new_tokens=128)
            end_time = time.time()
            
            elapsed = end_time - start_time
            total_time += elapsed
            successful_tests += 1
            
            # 토큰 수 계산
            tokens = len(tester.tokenizer.encode(answer)) if tester.tokenizer else len(answer.split())
            total_tokens += tokens
            
            print(f"   응답: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"   ⏱️  시간: {elapsed:.2f}초")
            print(f"   📝 토큰: {tokens}개")
            print(f"   🚀 토큰/초: {tokens/elapsed:.1f}")

        except Exception as e:
            print(f"   ❌ 테스트 실패: {e}")
            logger.error(f"벤치마크 테스트 {i} 실패: {e}")
            
    # 결과 요약
    print("\n📈 벤치마크 결과:")
    print(f"   성공한 테스트: {successful_tests}/{len(test_cases)}")
    if successful_tests > 0:
        avg_time = total_time / successful_tests
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        print(f"   총 시간: {total_time:.2f}초")
        print(f"   평균 응답 시간: {avg_time:.2f}초/요청")
        print(f"   총 생성 토큰: {total_tokens}개")
        print(f"   평균 생성 속도: {avg_tokens_per_sec:.1f} 토큰/초")
        
        # RTX 5060 성능 평가
        if avg_tokens_per_sec > 15:
            print("   🚀 성능: 우수 (RTX 5060 최적화 성공)")
        elif avg_tokens_per_sec > 10:
            print("   ⚡ 성능: 양호")
        else:
            print("   🐌 성능: 개선 필요 (메모리/설정 확인)")

def main():
    """메인 실행 함수"""
    print("="*60)
    print("  KoAlpaca 5.8B 모델 테스트 클라이언트 (RTX 5060 최적화)")
    print("="*60)
    
    tester = KoAlpacaModelTester()
    
    # GPU 정보 확인
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🔥 GPU: {gpu_name}")
        logger.info(f"🔥 GPU 메모리: {gpu_memory:.1f} GB")
    else:
        logger.warning("⚠️ CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    # 모델 연결 테스트
    if not test_model_connection(tester):
        print("\n❌ 모델 로딩에 실패했습니다.")
        print("💡 'python download_model.py'를 먼저 실행해주세요.")
        return

    print("\n실행할 작업을 선택하세요:")
    print("1. 대화형 챗봇 (Interactive Chat)")
    print("2. 성능 벤치마크 (Benchmark)")
    print("3. 단일 테스트 (Single Test)")
    choice = input("\n선택 (1-3, 기본값: 1): ").strip()

    try:
        if choice == "2":
            run_benchmark(tester)
        elif choice == "3":
            prompt = input("테스트할 질문을 입력하세요: ").strip()
            if prompt:
                print("\n🤖 응답:")
                answer = tester.generate_response(prompt)
                print(answer)
        else:
            run_interactive_chat(tester)
    
    except KeyboardInterrupt:
        print("\n\n👋 프로그램이 중단되었습니다.")
    
    finally:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU 메모리 정리 완료")

if __name__ == "__main__":
    main()
