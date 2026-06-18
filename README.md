# 📈 gpt-oss-trading-policy: LLM 기반 트레이딩 정책 파인튜닝 프로젝트

이 저장소는 룰 기반(Rule-based) 퀀트 투자 전략을 행동 모방(Behavioral Cloning) 기법을 통해 LLM(Large Language Model)에 이식하는 파인튜닝 파이프라인을 다룹니다. Apple Silicon 환경(M1/M2/M3 시리즈 Mac)에 최적화된 **MLX 라이브러리**를 사용하여 200억 매개변수 규모의 `gpt-oss-20b` MoE(Mixture of Experts) 모델을 LoRA 파인튜닝하는 안정화된 코드를 포함하고 있습니다.

---

## 📂 저장소 구조 (Repository Structure)

프로젝트 저장소는 전문적인 설계 구조에 맞추어 다음과 같이 데이터 수집, 가공, 학습 스크립트가 체계화되어 분리되어 있습니다.

```
gpt-oss-trading-policy/
├── data/
│   └── train_chat.jsonl      # OpenAI Chat Format으로 인코딩된 SFT 최종 학습 데이터셋
├── src/
│   ├── data_pipeline.py      # yfinance 기반 10개년 가격 수집 및 트레이딩 시뮬레이션 데이터 생성기
│   ├── preprocess.py         # raw 데이터를 MLX SFT 규격(OpenAI messages 포맷)으로 가공하는 도구
│   └── train.py              # MLX 메모리 누수 방지 및 trainable 파라미터 미분 최적화 반영 학습 스크립트
├── .gitignore
├── requirements.txt
└── README.md
```

### 📁 상세 파일 설명
*   **`src/data_pipeline.py`**: 9개 미국 기술주 역사적 시세를 파싱하여 단기/장기 이동평균 추세, 변동성, 유동성 지표에 기반한 룰 기반 포지션 타겟 결정을 자동 축적하는 파이프라인.
*   **`src/preprocess.py`**: 빌드된 원본 트레이딩 레코드를 `tokenizer.apply_chat_template`에 즉시 매핑 가능한 OpenAI 대화형 JSONL 데이터셋으로 전처리.
*   **`src/train.py`**: LoRA 가중치만 선별 미분 계산하는 `trainable_parameters()` 최적화, `mx.clear_cache()` 메모리 해제 장치를 장착하여 로컬 OOM을 원천 방어하는 메인 파인튜닝 파일.
*   **`data/train_chat.jsonl`**: 최종 생성 완료된 SFT 훈련 셋 (약 21,405개 샘플).

---

## 🧠 모델 가중치 (LoRA Adapters) 및 Hugging Face 연동

학습 결과로 산출되는 LoRA 어댑터 가중치(`adapters.npz` 파일, 약 98MB)는 Git 저장소의 크기 제한 및 청결한 관리를 위해 Hugging Face Model Hub에 분리 호스팅하고 있습니다.

*   **Hugging Face 저장소 링크**: [mlx-community/gpt-oss-20b-trading-lora-merged](https://huggingface.co/mlx-community/gpt-oss-20b-trading-lora-merged) (예시)
*   **어댑터 다운로드 및 병합**:
    ```python
    from mlx_lm import load
    # Hugging Face 허브에서 로라 병합 모델을 직접 다운로드하여 로드합니다.
    model, tokenizer = load("mlx-community/gpt-oss-20b-trading-lora-merged")
    ```

---

## ⚙️ 실행 방법 (Usage)

### 1. 의존성 패키지 설치
Apple Silicon 기기 혹은 로컬 파이썬 가상 환경에서 다음 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

### 2. 학습용 데이터 생성 및 전처리
```bash
# 1) 원본 train.jsonl 데이터셋 생성
python src/data_pipeline.py

# 2) MLX SFT용 train_chat.jsonl 변환 및 저장
python src/preprocess.py
```

### 3. SFT (LoRA 파인튜닝) 실행
```bash
# SFT 파인튜닝 시작 (학습 도중 log.csv에 Loss 기록 및 완료 시 lora_trading_policy_mlx/adapters.npz로 저장)
python src/train.py
```

---

## 🧠 핵심 기술적 특징 (Key Technical Features)

1.  **MoE 모델용 LoRA 적용 설계**:
    FFN 전문가(Experts) 영역 대신 모든 토큰이 경유하는 Self-Attention 레이어(`q_proj`, `v_proj`)만을 정밀 타격함으로써 학습 파라미터를 최소화하고 Mac Studio 수준의 VRAM에서 부드럽게 작동합니다.
2.  **MLX 지연 평가(Lazy Evaluation) 메모리 제어**:
    `mx.eval()`을 이용해 로스 연산을 GPU 디바이스에 즉시 바인딩 동기화하고 `mx.clear_cache()`로 주기적 메모리 청소를 실행하여 GPU VRAM OOM 현상을 원천 방지합니다.
3.  **Chat Format 데이터셋 연동**:
    `preprocess.py`를 통해 전처리된 Chat Format의 `messages` 구조를 `train.py` 내부 데이터 반복자에서 자동으로 감지해 `apply_chat_template`으로 통합 로드합니다.
