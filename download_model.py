# download_model.py

import os
import logging
import traceback
from pathlib import Path
from huggingface_hub import snapshot_download

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
BASE_MODELS_PATH = Path(__file__).parent / "models"
TARGET_MODEL_PATH = BASE_MODELS_PATH / MODEL_ID

def check_model_exists() -> bool:
    """모델이 이미 존재하는지 확인합니다."""
    config_file_path = TARGET_MODEL_PATH / "config.json"
    model_files = list(TARGET_MODEL_PATH.glob("*.safetensors")) if TARGET_MODEL_PATH.exists() else []
    
    exists = config_file_path.exists() and len(model_files) > 0
    if exists:
        logger.info(f"✅ 모델이 이미 '{TARGET_MODEL_PATH}'에 존재합니다.")
        logger.info(f"   - 설정 파일: {config_file_path}")
        logger.info(f"   - 모델 파일 수: {len(model_files)}")
    
    return exists

def download_model():
    """
    지정된 경로에 모델이 없으면 Hugging Face Hub에서 다운로드합니다.
    RTX 5060 호환성을 고려한 안전한 다운로드를 수행합니다.
    """
    logger.info("=" * 60)
    logger.info("      KoAlpaca 5.8B 모델 다운로드를 시작합니다.      ")
    logger.info("=" * 60)
    logger.info(f"모델 ID: {MODEL_ID}")
    logger.info(f"저장될 경로: {TARGET_MODEL_PATH}")
    logger.info(f"RTX 5060 호환성 설정 적용됨")
    logger.info("-" * 60)

    try:
        # 디렉토리 생성
        BASE_MODELS_PATH.mkdir(parents=True, exist_ok=True)
        TARGET_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        
        if check_model_exists():
            logger.info("다운로드를 건너뜁니다.")
            return True
        
        logger.info("모델 파일이 없습니다. 지금 다운로드를 시작합니다...")
        logger.warning("⚠️  모델 크기가 약 5.8GB이므로 몇 분에서 수십 분 정도 소요될 수 있습니다.")
        logger.info("📶 인터넷 연결 상태와 다운로드 속도를 확인해주세요.")
        
        # 다운로드 진행
        logger.info("🔄 Hugging Face Hub에서 다운로드 중...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=TARGET_MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py"],  # 필요한 파일만 다운로드
            ignore_patterns=["*.bin"],  # 이전 버전 파일 제외
        )
        
        # 다운로드 완료 확인
        if check_model_exists():
            logger.info("")
            logger.info("🎉" * 20)
            logger.info("🎉 모델 다운로드 완료!")
            logger.info(f"🎉 모델이 성공적으로 '{TARGET_MODEL_PATH}'에 저장되었습니다.")
            logger.info("🎉" * 20)
            
            # 다운로드된 파일 목록 표시
            model_files = list(TARGET_MODEL_PATH.glob("*"))
            logger.info(f"📁 다운로드된 파일 수: {len(model_files)}")
            for file in sorted(model_files)[:10]:  # 처음 10개 파일만 표시
                file_size = file.stat().st_size / (1024**2) if file.is_file() else 0
                logger.info(f"   - {file.name} ({file_size:.1f} MB)")
            
            if len(model_files) > 10:
                logger.info(f"   ... 및 {len(model_files) - 10}개 추가 파일")
            
            return True
        else:
            raise RuntimeError("다운로드는 완료되었지만 모델 파일을 찾을 수 없습니다.")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자에 의해 다운로드가 중단되었습니다.")
        logger.info("💡 resume_download=True 옵션으로 다음에 이어서 다운로드할 수 있습니다.")
        return False
        
    except Exception as e:
        logger.error("")
        logger.error("❌" * 20)
        logger.error(f"❌ 모델 다운로드 중 오류가 발생했습니다:")
        logger.error(f"❌ 오류 내용: {e}")
        logger.error("❌" * 20)
        logger.error("🔧 문제 해결 방법:")
        logger.error("   1. 인터넷 연결 상태를 확인해주세요")
        logger.error("   2. 디스크 용량이 충분한지 확인해주세요 (최소 6GB 필요)")
        logger.error("   3. Hugging Face Hub 접근 권한을 확인해주세요")
        logger.error("   4. 방화벽이나 프록시 설정을 확인해주세요")
        logger.error("")
        logger.error("🐛 상세 오류 정보:")
        traceback.print_exc()
        return False
        
    finally:
        logger.info("=" * 60)

def validate_model():
    """다운로드된 모델의 무결성을 간단히 검증합니다."""
    logger.info("🔍 모델 파일 무결성 검증 중...")
    
    if not check_model_exists():
        logger.error("❌ 모델이 존재하지 않습니다.")
        return False
    
    try:
        # 필수 파일들 확인
        required_files = ["config.json"]
        safetensors_files = list(TARGET_MODEL_PATH.glob("*.safetensors"))
        
        missing_files = []
        for file_name in required_files:
            if not (TARGET_MODEL_PATH / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"❌ 필수 파일이 누락되었습니다: {missing_files}")
            return False
        
        if len(safetensors_files) == 0:
            logger.error("❌ 모델 가중치 파일(.safetensors)이 없습니다.")
            return False
        
        logger.info(f"✅ 모델 검증 완료! 가중치 파일 {len(safetensors_files)}개 확인됨")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 검증 중 오류 발생: {e}")
        return False

def get_model_info():
    """다운로드된 모델의 정보를 반환합니다."""
    if not check_model_exists():
        return None
    
    try:
        import json
        config_path = TARGET_MODEL_PATH / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_files = list(TARGET_MODEL_PATH.glob("*"))
        total_size = sum(f.stat().st_size for f in model_files if f.is_file())
        
        info = {
            "model_id": MODEL_ID,
            "path": str(TARGET_MODEL_PATH),
            "model_type": config.get("model_type", "unknown"),
            "vocab_size": config.get("vocab_size", 0),
            "total_files": len(model_files),
            "total_size_gb": total_size / (1024**3),
            "architecture": config.get("architectures", ["unknown"])[0] if config.get("architectures") else "unknown"
        }
        
        return info
        
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: {e}")
        return None

# --- 테스트 코드 ---
if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("  KoAlpaca 5.8B 모델 다운로더 단독 실행 테스트 시작")
        print("="*60)
        
        # RTX 5060 환경 정보 출력
        logger.info("🖥️  RTX 5060 호환성 설정이 적용되었습니다.")
        logger.info(f"📁 대상 경로: {TARGET_MODEL_PATH}")
        
        # 모델 다운로드 실행
        success = download_model()
        
        if success:
            # 모델 검증
            if validate_model():
                # 모델 정보 출력
                info = get_model_info()
                if info:
                    logger.info("\n📋 모델 정보:")
                    logger.info(f"   - 모델 타입: {info['model_type']}")
                    logger.info(f"   - 아키텍처: {info['architecture']}")
                    logger.info(f"   - 어휘 크기: {info['vocab_size']:,}")
                    logger.info(f"   - 전체 크기: {info['total_size_gb']:.2f} GB")
                    logger.info(f"   - 파일 수: {info['total_files']}")
                
                print("\n" + "🎉"*20)
                print("🎉 테스트 성공! 모델이 정상적으로 다운로드되었습니다.")
                print("🎉 이제 train.py에서 로컬 모델을 사용할 수 있습니다!")
                print("🎉"*20)
            else:
                print("\n❌ 모델 검증에 실패했습니다.")
        else:
            print("\n❌ 모델 다운로드에 실패했습니다.")
        
    except Exception as e:
        print(f"\n🔥🔥🔥 테스트 실패! 오류가 발생했습니다: {e} 🔥🔥🔥")
        traceback.print_exc()
    
    finally:
        print("\n" + "="*60)
        print("  테스트 종료")
        print("="*60)