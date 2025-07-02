# GRI-PEFT: ESG 보고서 생성을 위한 LoRA 파인튜닝 프로젝트

이 프로젝트는 **KoAlpaca-Polyglot-5.8B** 모델을 **LoRA(Low-Rank Adaptation)** 방식으로 파인튜닝하여, **GRI(Global Reporting Initiative) 기준에 맞는 ESG 보고서를 자동 생성**하는 모델을 개발합니다.

## 📋 목차
- [프로젝트 개요](#프로젝트-개요)
- [환경 설정](#환경-설정)
- [모델 다운로드](#모델-다운로드)
- [데이터 준비](#데이터-준비)
- [모델 훈련](#모델-훈련)
- [모델 테스트](#모델-테스트)
- [결과](#결과)
- [파일 구조](#파일-구조)

## 🎯 프로젝트 개요

### 목적
- GRI 기준에 따른 구조화된 데이터를 전문적인 한국어 ESG 보고서 문단으로 자동 변환
- 기업의 지속가능성 보고서 작성 업무 효율화
- LoRA 기법을 활용한 효율적인 대규모 언어 모델 파인튜닝

### 주요 특징
- **모델**: KoAlpaca-Polyglot-5.8B (5.8B 파라미터)
- **훈련 방식**: LoRA (Low-Rank Adaptation)
- **양자화**: 4bit NF4 양자화 (메모리 효율성)
- **하드웨어**: NVIDIA RTX 5060 최적화
- **데이터**: 116개 GRI 표준 기반 ESG 보고서 샘플

## 🛠️ 환경 설정

### 시스템 요구사항
- **OS**: Windows 10/11
- **GPU**: NVIDIA RTX 5060 (8GB VRAM)
- **Python**: 3.12.7 (권장)
- **CUDA**: 12.8+

### 의존성 설치

1. **Conda 환경 생성**
```bash
conda create -n gri-env python=3.12.7
conda activate gri-env
```

2. **PyTorch 설치**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3. **필수 라이브러리 설치**
```bash
pip install -r requirements.txt
```

### requirements.txt
```
transformers>=4.36.0
peft>=0.7.0
trl>=0.7.0
datasets>=2.14.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
langchain-community>=0.0.10
```

## 📥 모델 다운로드

KoAlpaca-Polyglot-5.8B 모델을 로컬에 다운로드합니다:

```bash
python download_model.py
```

모델은 `models/beomi/KoAlpaca-Polyglot-5.8B/` 디렉토리에 저장됩니다.

## 📊 데이터 준비

### 데이터 형식
훈련 데이터는 JSONL 형식으로 다음과 같은 구조를 가집니다:

```json
{
  "prompt": "You are an expert ESG report writer. Based on the following structured data, synthesize the information into a single, cohesive, and professional Korean paragraph for a sustainability report.\n\n### Data:\n{...}\n\n### Polished Report Paragraph:",
  "completion": "전문적인 한국어 ESG 보고서 문단"
}
```

### 데이터 파일
- **원본 데이터**: `gri_data_fixed_corrected.jsonl` (116개 샘플)
- **훈련용 데이터**: 각 JSON 객체가 줄바꿈으로 구분된 올바른 JSONL 형식

## 🚀 모델 훈련

### 기본 훈련 명령어
```bash
python train_rola.py --dataset_path gri_data_fixed_corrected.jsonl
```

### 고급 옵션
```bash
python train_rola.py \
  --dataset_path gri_data_fixed_corrected.jsonl \
  --output_dir ./lora_results_custom \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_seq_len 2048
```

### 훈련 파라미터
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--dataset_path` | 필수 | 훈련 데이터 JSONL 파일 경로 |
| `--output_dir` | `./lora_results_rtx5060` | 결과 저장 디렉토리 |
| `--epochs` | `3` | 훈련 에포크 수 |
| `--batch_size` | `8` | 배치 크기 (RTX 5060: 4-6 권장) |
| `--learning_rate` | `2e-4` | 학습률 |
| `--max_seq_len` | `2048` | 최대 시퀀스 길이 |

### LoRA 설정
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Target Modules**: query_key_value
- **훈련 가능한 파라미터**: 7,340,032개 (전체의 0.1246%)

### 4bit 양자화 설정
- **양자화 타입**: NF4 (NormalFloat4)
- **Double Quantization**: 활성화 (추가 메모리 절약)
- **Compute dtype**: bfloat16
- **메모리 사용량**: 약 4GB (양자화 전 대비 75% 절약)

**양자화의 장점:**
- ✅ **메모리 효율성**: 5.8B 모델을 8GB GPU에서 훈련 가능
- ✅ **훈련 속도**: 메모리 대역폭 감소로 빠른 학습
- ✅ **정확도 유지**: NF4 양자화로 성능 손실 최소화

**코드 예시:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4bit 로딩 활성화
    bnb_4bit_use_double_quant=True,       # 이중 양자화 (더 많은 메모리 절약)
    bnb_4bit_quant_type="nf4",            # NormalFloat4 사용
    bnb_4bit_compute_dtype=torch.bfloat16 # 계산용 데이터 타입
)
```

## 🧪 모델 테스트

훈련된 모델을 테스트하려면:

```bash
python test_custom_prompt.py
```

### 테스트 예시

**입력 프롬프트**:
```json
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
```

**생성된 출력**:
> 당사는 산업재해 예방 및 보건환경 개선을 위해 다음과 같은 노력을 하고 있습니다...

## 📈 결과

### 훈련 성과
- **데이터셋**: 116개 GRI 기반 ESG 샘플
- **훈련 시간**: 약 10-15분 (RTX 5060)
- **에포크**: 3회 (총 126 스텝)
- **최종 Loss**: 수렴됨

### 모델 성능
- ✅ **언어**: 자연스러운 한국어 생성
- ✅ **형식**: 전문적인 ESG 보고서 스타일
- ✅ **내용**: 관련 키워드 및 개념 적절히 반영
- ✅ **일관성**: 안정적이고 논리적인 문단 구성

### 최적 생성 파라미터
- **Temperature**: 0.5 (일관된 답변용)
- **Top-p**: 0.95
- **Max Length**: 800 토큰
- **Repetition Penalty**: 1.1

## 📁 파일 구조

```
gri-peft/
├── README.md                     # 프로젝트 문서
├── requirements.txt              # 의존성 목록
├── download_model.py             # 모델 다운로드 스크립트
├── train_rola.py                 # LoRA 훈련 스크립트
├── test_custom_prompt.py         # 모델 테스트 스크립트
├── fix_jsonl.py                  # JSONL 형식 수정 도구
├── gri_data_fixed_corrected.jsonl # 훈련 데이터 (116개 샘플)
├── models/
│   └── beomi/
│       └── KoAlpaca-Polyglot-5.8B/  # 베이스 모델
├── lora_corrected/
│   ├── checkpoint-14/            # 중간 체크포인트
│   ├── checkpoint-29/            # 중간 체크포인트  
│   ├── checkpoint-42/            # 중간 체크포인트
│   └── final_lora_adapter/       # 최종 LoRA 어댑터
└── lora_results_*/               # 기타 훈련 결과
```

## 🔧 문제 해결

### 일반적인 문제들

1. **JSONL 파싱 오류**
   ```bash
   python fix_jsonl.py  # JSONL 형식 자동 수정
   ```

2. **CUDA 메모리 부족**
   - `batch_size`를 4 이하로 낮춤
   - `gradient_accumulation_steps` 증가

3. **bitsandbytes 호환성**
   ```bash
   pip uninstall bitsandbytes
   pip install bitsandbytes --no-cache-dir
   ```

4. **양자화 관련 오류**
   - CUDA 버전 불일치: `nvcc --version`으로 확인
   - 지원되지 않는 GPU: RTX 20 시리즈 이상 필요
   - 메모리 할당 실패: `torch.cuda.empty_cache()` 실행

## 🚀 사용 사례

이 모델은 다음과 같은 용도로 활용할 수 있습니다:

- **기업 ESG 보고서 자동화**: GRI 기준 데이터를 전문적인 보고서 문단으로 변환
- **지속가능성 컨설팅**: 표준화된 ESG 콘텐츠 생성
- **교육 및 연구**: ESG 보고 모범 사례 학습

## 📞 기술 지원

프로젝트 관련 문의사항이나 이슈가 있으시면 GitHub Issues를 통해 제보해주세요.

---

**개발 환경**: Windows 11 + NVIDIA RTX 5060 + Python 3.12.7  
**라이선스**: MIT License  
**마지막 업데이트**: 2024년 12월 