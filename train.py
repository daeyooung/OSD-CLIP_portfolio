# coding=utf-8
"""OSD-CLIP 학습 / open-set 평가 진입점 (Dassl 기반).

    # 학습 (마지막 epoch 체크포인트 저장)
    python train.py --config-file configs/trainers/osd_clip_vit_l14.yaml \
        --dataset-config-file configs/datasets/mvtec_pkl.yaml --output-dir results/run --seed 0 \
        DATASET.PKL_PATH mvtec_ac_pkl/metal_nut_seed0_unknown_bent.pkl DATASET.OBJECT_NAME "metal nut" \
        TRAINER.OSDCLIP.SOFTMASK_DIR <softmask dir>

    # open-set 평가 (저장된 마지막 epoch 프롬프트 사용)
    python train.py ... --open-others-test --model-dir results/run [--load-epoch 100]
"""
import argparse
import time

import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger

import datasets.mvtec_pkl   # noqa: F401  (DATASET_REGISTRY 등록)
import trainers.osd_clip    # noqa: F401  (TRAINER_REGISTRY 등록)


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    cfg.DATASET.PKL_PATH = ""          # mvtec_ac_pkl/<category>_seed0_unknown_<defect>.pkl
    cfg.DATASET.OBJECT_NAME = ""       # 프롬프트에 들어갈 객체 이름 (예: "metal nut")
    cfg.DATASET.FEW_SHOT_K = 16        # 클래스당 학습 이미지 수 (0 = 전체)

    cfg.TRAINER.OSDCLIP = CN()
    cfg.TRAINER.OSDCLIP.N_CTX = 16                 # L : prompt context 토큰 수
    cfg.TRAINER.OSDCLIP.NUM_NEG_PROMPTS = 30       # M : unknown prompt 수
    cfg.TRAINER.OSDCLIP.LAMBDA = 5.0               # λ : 정상 영역 손실 가중치
    cfg.TRAINER.OSDCLIP.TOPK = 20                  # Q : 패치 선택 수 (정상 영역 / threshold 미달 fallback)
    cfg.TRAINER.OSDCLIP.T = 1.0                    # softmax 온도
    cfg.TRAINER.OSDCLIP.SOFTMASK_THRESHOLD = 0.9   # τd : 결함 패치 선택 softmask 임계값
    cfg.TRAINER.OSDCLIP.OLS = 0.001                # ε : unknown label smoothing
    cfg.TRAINER.OSDCLIP.NORMAL_CROP_ROI_SIZE = 224  # 정상 영역 crop 크기 (원본 해상도 px)
    cfg.TRAINER.OSDCLIP.SOFTMASK_DIR = ""          # 카테고리별 softmask 폴더
    cfg.TRAINER.OSDCLIP.ALPHA_CLIP_CKPT = "./checkpoints/clip_l14_grit20m_fultune_2xe.pth"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_file(args.config_file)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SEED = args.seed
    cfg.TRAINER.NAME = "OSDCLIP"
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("** Config **")
    print(cfg)
    print("** System info **\n" + collect_env_info() + "\n")

    trainer = build_trainer(cfg)

    if args.open_others_test:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.open_test()
        return

    start = time.time()
    trainer.train()
    print(f"total training time: {time.time() - start:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./configs/trainers/osd_clip_vit_l14.yaml")
    parser.add_argument("--dataset-config-file", type=str, default="./configs/datasets/mvtec_pkl.yaml")
    parser.add_argument("--output-dir", type=str, default="./results/debug")
    parser.add_argument("--seed", type=int, default=0, help="음수면 seed 고정 안 함")
    parser.add_argument("--open-others-test", action="store_true", help="학습 없이 open-set 평가만 수행")
    parser.add_argument("--model-dir", type=str, default="", help="평가 시 체크포인트가 있는 output 디렉토리")
    parser.add_argument("--load-epoch", type=int, default=None, help="평가에 쓸 epoch (기본: OPTIM.MAX_EPOCH)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="cfg 덮어쓰기 (KEY VALUE ...)")
    main(parser.parse_args())
