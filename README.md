# OSD-CLIP: Anomaly-Aware Local Prompt Learning for Few-shot Open-Set Industrial Defect Recognition

산업 결함 이미지에서 **학습한 결함 유형(known)은 분류하고, 학습에 없던 결함 유형(unknown)은 탐지**하는
few-shot open-set 결함 인식 방법의 PyTorch 구현과 MVTec AC 실험 결과입니다.

- **문제**: MVTec AC 14개 카테고리에서 결함 유형 하나를 unknown 으로 두고 나머지를 known 으로 학습하는 **64개 open-set 시나리오**.
  known 클래스당 1 / 4 / 16 장만 학습(few-shot)
- **방법**: frozen Alpha-CLIP ViT-L/14 위에서 **프롬프트(context 토큰)만 학습**.
  PatchCore anomaly map(softmask)으로 결함 영역을 잘라 Alpha-CLIP 의 alpha 채널로 주고, 결함 패치는 known prompt 로,
  정상 영역 패치는 M 개의 unknown prompt 로 학습. 추론은 패치 단위 softmax 의 평균(MSP)으로 known 분류와 unknown 탐지를 동시에 수행
- **재현성**: seed 고정(few-shot 샘플·프롬프트 초기화·정상 영역 샘플링), 모델 선택 없이 **마지막 epoch 프롬프트로 평가**,
  모든 하이퍼파라미터는 [`configs/`](configs) 와 [`train_eval.sh`](train_eval.sh) 에 고정.

## 결과 (MVTec AC, 64 시나리오 × seed 0/1/2 = 192 run, 평균 ± seed 표준편차)

| shot | Acc_Known | AUROC | OSCR |
|:---:|:---:|:---:|:---:|
| 1  | 69.57 ± 4.51 | 68.10 ± 1.29 | 51.27 ± 2.43 |
| 4  | 86.62 ± 0.95 | 78.03 ± 1.39 | 70.58 ± 1.87 |
| 16 | 91.04 ± 0.33 | 82.16 ± 0.90 | 77.31 ± 0.89 |

Acc_Known = known 결함 분류 정확도, AUROC = known/unknown 구분, OSCR = 분류와 unknown 탐지를 함께 보는 open-set 지표.

16-shot 카테고리별 결과와 baseline 비교(ARPL, PROSER, NAPG, Local-Prompt, UTL: 같은 Alpha-CLIP 백본·데이터 분할·few-shot 샘플)는
[`results_summary/`](results_summary) 에 CSV 로 정리했습니다. baseline 방법의 코드는 이 저장소에 포함하지 않습니다.

| 16-shot | Acc_Known | AUROC | OSCR |
|---|:---:|:---:|:---:|
| **OSD-CLIP (ours)** | **91.04** | **82.16** | **77.31** |
| UTL | 84.76 | 72.43 | 65.38 |
| PROSER | 84.09 | 70.82 | 63.47 |
| ARPL | 83.28 | 70.04 | 62.15 |
| Local-Prompt | 76.83 | 55.92 | 42.57 |
| NAPG | 74.41 | 59.28 | 46.57 |

## 저장소 구성

**직접 작성한 코드**

| 파일 | 내용 |
|---|---|
| `trainers/osd_clip.py` | 모델·학습·평가 전체: Alpha-CLIP 패치 특징(V-V attention, 4-stage), prompt learner, 결함/정상 영역 손실, MSP 추론, open-set 지표 |
| `trainers/data_wrapper.py` | softmask 로드, top-1 component 기반 결함 영역 crop(ρ = 0.10 / 0.20, 9 view), 정상 영역 crop, Dataset |
| `datasets/mvtec_pkl.py` | pkl 분할 파일 → known/unknown 분리, seed 종속 few-shot 샘플링 |
| `mvtec_ac_pkl/` | 64개 시나리오의 train/test 분할 (`<category>_seed0_unknown_<defect>.pkl`) |
| `train.py`, `configs/` | yacs 설정과 진입점 (학습 / `--open-others-test` 평가) |
| `train_eval.sh` | 시나리오마다 학습 → 마지막 epoch 평가 → 카테고리별 집계 |
| `aggregate_by_category.py`, `check_setup.py` | 결과 집계(pandas), 환경 점검 |
| `tools/softmask/make_softmask.py` | PatchCore 3-model ensemble anomaly map → 정규화·blur → softmask `.npy` 생성 |

**외부 코드**: `Dassl.pytorch/` ([KaiyangZhou/Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), MIT; open-set 지표 함수 추가),
`clip/` ([OpenAI CLIP](https://github.com/openai/CLIP), MIT, 토크나이저), Alpha-CLIP 은 [SunzeY/AlphaCLIP](https://github.com/SunzeY/AlphaCLIP) 패키지 사용.

## 학습 설정 (코드에 고정)

| 항목 | 값 |
|---|---|
| 백본 | Alpha-CLIP ViT-L/14 (frozen), 입력 224×224, 증강 없음 |
| 학습 파라미터 | known context L=16 (공유) + unknown prompt M=30 × L=16 |
| 결함 영역 | softmask top-1 component 기준 ρ=0.10 shift 4 view + ρ=0.20 center/shift 5 view (이미지당 9) |
| 정상 영역 | 이미지당 9개, softmask ≤ 0.5 인 224px 정사각, run 시작 시 고정 생성, alpha = 0 |
| 손실 | L_defect (softmask ≥ τd=0.9 패치, unknown ε=0.001 smoothing) + λ=5.0 · L_normal (GT class 유사도 top-Q=20 패치 → unknown 균등) |
| 최적화 | SGD lr 0.002 cosine, momentum 0.9, wd 5e-4, warm-up 1 epoch (1e-5), **batch 16**, **100 epoch**, fp32 |
| 평가 | 검증셋·early stopping 없음, **마지막 epoch** 프롬프트. 테스트 입력 ρ=0.20 중심 영역 1개, τd 동일 적용 |

## 환경 설정

검증 환경: Python 3.10, PyTorch 2.9.1 (CUDA 12.8).

```bash
pip install -r requirements.txt           # PyTorch 는 CUDA 버전에 맞게 먼저 설치
pip install -e Dassl.pytorch              # (train_eval.sh 는 PYTHONPATH 로 저장소의 Dassl 을 직접 쓰므로 생략 가능)
pip install git+https://github.com/SunzeY/AlphaCLIP.git

# Alpha-CLIP 가중치 (ViT-L/14, AlphaCLIP model zoo 의 clip_l14_grit20m_fultune_2xe.pth)
mkdir -p checkpoints && mv clip_l14_grit20m_fultune_2xe.pth checkpoints/
```

## Data

저장소 루트에 아래 두 가지를 둡니다 (심볼릭 링크 가능). 용량 문제로 저장소에는 포함하지 않았습니다.

1. **MVTec AC** ([VELM](https://github.com/Sassanmtr/VELM) 에서 배포, MVTec AD 테스트 이미지를 결함 분류용으로 재정리한 데이터셋) → `Dataset/mvtec_ac/<category>/test/<class>/*.png`
2. **softmask** (PatchCore anomaly map, `.npy` 0~1) → `alphaclip_softmask_method3_robust_0.5_blur/alphaclip_masks_method3_robust_blur_ac_<category>/<class>/softmask/<image>.npy`
   `tools/softmask/make_softmask.py` 로 생성합니다. MVTec AD 원본([공식 페이지](https://www.mvtec.com/company/research/datasets/mvtec-ad))의
   `train/good` 정상 이미지(MVTec AC 는 MVTec AD 와 같은 학습 이미지를 쓰며 배포본에는 포함되지 않음)와
   [patchcore-inspection](https://github.com/amazon-science/patchcore-inspection) 이 필요합니다.

   ```bash
   git clone https://github.com/amazon-science/patchcore-inspection external/patchcore-inspection
   pip install faiss-cpu pretrainedmodels scikit-image
   python tools/softmask/make_softmask.py --dataset_root Dataset/mvtec_ac --train_root Dataset/mvtec_ad --gpu 0
   ```

```bash
python check_setup.py      # 패키지 / GPU / 데이터 / softmask / 체크포인트 점검
```

## 실행

```bash
# 동작 확인: 시나리오 1개 학습 + 평가
PKL_FILTER=carpet_seed0_unknown_color bash train_eval.sh

# 64개 시나리오, seed 0. 논문 설정은 SEEDS="0 1 2"
bash train_eval.sh

# 결과 집계만 다시
python aggregate_by_category.py --results_dir results/osd_clip_fewshot16/ep100
```

결과는 `results/osd_clip_fewshot16/ep100/<scenario>/seed<S>/` 에 `open_test_results_open_others_msp.csv`(지표), `log.txt`,
confusion matrix, 입력 샘플 이미지로 저장되고, `aggregated_by_category.csv` 에 카테고리별 평균이 정리됩니다.
`FEW_SHOT_K`, `EPOCHS`, `SEEDS`, `GPU_ID` 는 환경변수로 바꿀 수 있습니다.
