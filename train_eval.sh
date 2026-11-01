#!/usr/bin/env bash
# 64개 설정(카테고리 x unknown 결함) 마다 학습 -> 마지막 epoch 체크포인트로 open-set 평가 -> 집계.
#
#   bash train_eval.sh                                        # 64개 설정, seed 0, 100 epoch
#   PKL_FILTER=carpet_seed0_unknown_color bash train_eval.sh  # 설정 1개만 (동작 확인, GPU 1장 약 15분)
#   SEEDS="0 1 2" bash train_eval.sh                          # 논문과 같은 3 seed
#   MODEL_SRC=/path/to/results bash train_eval.sh             # 학습 생략, 기존 체크포인트 재평가
#
# 환경변수: GPU_ID(0) EPOCHS(100) FEW_SHOT_K(16) SEEDS("0") PKL_FILTER ALPHA_CLIP_CKPT MODEL_SRC
set -euo pipefail

PROJ="$(cd "$(dirname "$0")"; pwd)"
cd "${PROJ}"
export PYTHONPATH="${PROJ}/Dassl.pytorch:${PROJ}:${PYTHONPATH:-}"   # 저장소에 포함된 Dassl 사용
PKL_DIR="${PROJ}/mvtec_ac_pkl"
SOFTMASK_ROOT="${PROJ}/alphaclip_softmask_method3_robust_0.5_blur"
TRAINER_CFG="configs/trainers/osd_clip_vit_l14.yaml"
DATASET_CFG="configs/datasets/mvtec_pkl.yaml"

GPU_ID="${GPU_ID:-0}"
FEW_SHOT_K="${FEW_SHOT_K:-16}"
read -r -a EPOCHS <<< "${EPOCHS:-100}"
read -r -a SEEDS <<< "${SEEDS:-0}"
ALPHA_CLIP_CKPT="${ALPHA_CLIP_CKPT:-${PROJ}/checkpoints/clip_l14_grit20m_fultune_2xe.pth}"
MODEL_SRC="${MODEL_SRC:-}"                       # 지정 시 <MODEL_SRC>/ep<E>/<pkl>/seed<S>/ 의 체크포인트로 평가만
RESULT_ROOT="${PROJ}/results/osd_clip_fewshot${FEW_SHOT_K}${SWEEP_TAG:-}"

if [[ ! -f "${ALPHA_CLIP_CKPT}" ]]; then
  echo "[ERROR] Alpha-CLIP checkpoint not found: ${ALPHA_CLIP_CKPT} (README '환경 설정' 참고)"; exit 1
fi

mapfile -t pkls < <(ls -1 "${PKL_DIR}"/*.pkl | sort)
if [[ -n "${PKL_FILTER:-}" ]]; then
  mapfile -t pkls < <(printf '%s\n' "${pkls[@]}" | grep -- "${PKL_FILTER}" || true)
fi
[[ ${#pkls[@]} -eq 0 ]] && { echo "[ERROR] no pkl matched PKL_FILTER=${PKL_FILTER:-}"; exit 1; }
echo "pkl: ${#pkls[@]} | seeds: ${SEEDS[*]} | epochs: ${EPOCHS[*]} | few-shot K=${FEW_SHOT_K} | results: ${RESULT_ROOT}"

for MAX_EPOCH in "${EPOCHS[@]}"; do
for pkl in "${pkls[@]}"; do
  pkl_base="$(basename "${pkl}" .pkl)"
  CATEGORY="${pkl_base%%_*}"                                   # 파일명 첫 단어 (metal_nut -> metal)
  SOFTMASK_DIR="${SOFTMASK_ROOT}/alphaclip_masks_method3_robust_blur_ac_${CATEGORY}"
  [[ "${CATEGORY}" == "metal" ]] && CATEGORY="metal_nut"
  OBJECT_NAME="${CATEGORY//_/ }"

  OPTS=(
    DATASET.PKL_PATH "${pkl}"
    DATASET.FEW_SHOT_K "${FEW_SHOT_K}"
    DATASET.OBJECT_NAME "${OBJECT_NAME}"
    OPTIM.MAX_EPOCH "${MAX_EPOCH}"
    TRAINER.OSDCLIP.SOFTMASK_DIR "${SOFTMASK_DIR}"
    TRAINER.OSDCLIP.ALPHA_CLIP_CKPT "${ALPHA_CLIP_CKPT}"
  )

  for seed in "${SEEDS[@]}"; do
    outdir="${RESULT_ROOT}/ep${MAX_EPOCH}/${pkl_base}/seed${seed}"
    mkdir -p "${outdir}"
    if [[ -f "${outdir}/open_test_results_open_others_msp.csv" ]]; then
      echo "[SKIP] done: ${outdir}"; continue
    fi
    echo "============================================"
    echo "pkl: ${pkl_base} | object: ${OBJECT_NAME} | seed: ${seed} | epoch: ${MAX_EPOCH}"
    echo "============================================"

    if [[ -z "${MODEL_SRC}" ]]; then
      model_dir="${outdir}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
        --config-file "${TRAINER_CFG}" --dataset-config-file "${DATASET_CFG}" \
        --output-dir "${outdir}" --seed "${seed}" "${OPTS[@]}"
    else
      model_dir="${MODEL_SRC}/ep${MAX_EPOCH}/${pkl_base}/seed${seed}"
    fi

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
      --config-file "${TRAINER_CFG}" --dataset-config-file "${DATASET_CFG}" \
      --output-dir "${outdir}" --seed "${seed}" \
      --open-others-test --model-dir "${model_dir}" --load-epoch "${MAX_EPOCH}" "${OPTS[@]}"
  done
done
  python aggregate_by_category.py --results_dir "${RESULT_ROOT}/ep${MAX_EPOCH}"
done

echo "All done. Results: ${RESULT_ROOT}"
