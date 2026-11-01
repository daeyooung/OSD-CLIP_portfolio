# coding=utf-8
"""PatchCore anomaly map -> softmask 생성 (학습 코드가 읽는 .npy 파일을 만든다).

파이프라인 (카테고리마다):
  1. train/good 이미지로 PatchCore 메모리 뱅크 구축 (patchcore-inspection, 3-model ensemble:
     WideResNet101 + ResNeXt101 + DenseNet201, layer2/3, coreset 1%)
  2. test/<defect>/ 이미지의 anomaly map 예측 -> 모델별 global min-max 정규화 -> 평균
  3. 이미지별 robust normalize (5 ~ 99.5 percentile) -> Gaussian blur (5x5) -> [0,1] clip
  4. <out_root>/alphaclip_masks_method3_robust_blur_ac_<cat>/<defect>/softmask/<stem>.npy (+ .png)

사용법:
  git clone https://github.com/amazon-science/patchcore-inspection external/patchcore-inspection
  pip install faiss-gpu   # 또는 faiss-cpu
  python tools/softmask/make_softmask.py --dataset_root Dataset/mvtec_ac --train_root Dataset/mvtec_ad --gpu 0
  python tools/softmask/make_softmask.py --categories bottle carpet --single   # WideResNet50 단일 모델 (빠름)
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PATCHCORE = os.path.join(REPO_ROOT, "external", "patchcore-inspection", "src")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE, IMAGESIZE = 256, 224

# patchcore-inspection run_patchcore.py 의 IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1 설정
ENSEMBLE_CFG = [
    dict(backbone_name="wideresnet101", layers=["layer2", "layer3"], pretrain_embed_dim=1024, target_embed_dim=384, coreset_p=0.01),
    dict(backbone_name="resnext101", layers=["layer2", "layer3"], pretrain_embed_dim=1024, target_embed_dim=384, coreset_p=0.01),
    dict(backbone_name="densenet201", layers=["features.denseblock2", "features.denseblock3"], pretrain_embed_dim=1024, target_embed_dim=384, coreset_p=0.01),
]
# IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1 설정
SINGLE_CFG = [
    dict(backbone_name="wideresnet50", layers=["layer2", "layer3"], pretrain_embed_dim=1024, target_embed_dim=1024, coreset_p=0.1),
]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class ImageFolderDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = T.Compose([T.Resize(RESIZE), T.CenterCrop(IMAGESIZE), T.ToTensor(),
                                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.paths[i]).convert("RGB"))
        return {"image": img, "image_path": self.paths[i],
                "mask": torch.zeros(1, IMAGESIZE, IMAGESIZE), "is_anomaly": 1}


def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in IMG_EXTS)


def build_patchcore(pc_mod, cfg, device, num_workers):
    backbone = pc_mod.backbones.load(cfg["backbone_name"])
    backbone.name, backbone.seed = cfg["backbone_name"], None
    pc = pc_mod.patchcore.PatchCore(device)
    pc.load(backbone=backbone, layers_to_extract_from=cfg["layers"], device=device,
            input_shape=(3, IMAGESIZE, IMAGESIZE),
            pretrain_embed_dimension=cfg["pretrain_embed_dim"], target_embed_dimension=cfg["target_embed_dim"],
            patchsize=3, anomaly_score_num_nn=1,
            featuresampler=pc_mod.sampler.ApproximateGreedyCoresetSampler(percentage=cfg["coreset_p"], device=device),
            nn_method=pc_mod.common.FaissNN(on_gpu=False, num_workers=num_workers))
    return pc


def robust_normalize(m, lower_p=5, upper_p=99.5):
    lo, hi = np.percentile(m, lower_p), np.percentile(m, upper_p)
    if hi - lo < 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    return np.clip((m - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def blur(m, ksize=5):
    return np.clip(cv2.GaussianBlur(m, (ksize, ksize), sigmaX=0), 0.0, 1.0).astype(np.float32)


def process_category(pc_mod, category, args, device):
    cat_dir = os.path.join(args.dataset_root, category)
    train_good = os.path.join(args.train_root or args.dataset_root, category, "train", "good")
    test_dir = os.path.join(cat_dir, "test")
    if not os.path.isdir(train_good) or not os.path.isdir(test_dir):
        print(f"  [SKIP] {category}: train/good 또는 test 폴더 없음")
        return

    defect_classes = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)) and d != "good")
    test_paths, test_cls = [], []
    for c in defect_classes:
        p = list_images(os.path.join(test_dir, c))
        test_paths += p
        test_cls += [c] * len(p)
    if not test_paths:
        print(f"  [SKIP] {category}: 결함 이미지 없음")
        return

    loader_kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(ImageFolderDataset(list_images(train_good)), **loader_kw)
    test_loader = DataLoader(ImageFolderDataset(test_paths), **loader_kw)

    cfgs = SINGLE_CFG if args.single else ENSEMBLE_CFG
    per_model = []
    for i, cfg in enumerate(cfgs, 1):
        print(f"  [{i}/{len(cfgs)}] {cfg['backbone_name']}: fit ({len(train_loader.dataset)} imgs) -> predict ({len(test_paths)} imgs)")
        pc = build_patchcore(pc_mod, cfg, device, args.num_workers)
        pc.fit(train_loader)
        _, masks, _, _ = pc.predict(test_loader)          # list of [H, W] (RescaleSegmentor 적용됨)
        per_model.append(np.array(masks))
        del pc
        torch.cuda.empty_cache()

    segs = np.stack(per_model)                                             # [M, N, H, W]
    flat = segs.reshape(len(segs), -1)
    lo, hi = flat.min(1).reshape(-1, 1, 1, 1), flat.max(1).reshape(-1, 1, 1, 1)
    segs = (segs - lo) / (hi - lo + 1e-8)                                  # 모델별 global min-max
    avg = segs.mean(0)                                                     # [N, H, W]

    out_cat = os.path.join(args.out_root, f"alphaclip_masks_method3_robust_blur_ac_{'metal' if category == 'metal_nut' else category}")
    for m, path, c in zip(avg, test_paths, test_cls):
        sm = blur(robust_normalize(m))
        stem = os.path.splitext(os.path.basename(path))[0]
        sm_dir = os.path.join(out_cat, c, "softmask")
        os.makedirs(sm_dir, exist_ok=True)
        np.save(os.path.join(sm_dir, f"{stem}.npy"), sm)
        cv2.imwrite(os.path.join(sm_dir, f"{stem}.png"), (sm * 255).astype(np.uint8))
    print(f"  saved {len(test_paths)} softmasks -> {out_cat}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", default=os.path.join(REPO_ROOT, "Dataset", "mvtec_ac"))
    ap.add_argument("--train_root", default=None, help="train/good 이미지가 다른 위치에 있을 때 (기본: dataset_root)")
    ap.add_argument("--out_root", default=os.path.join(REPO_ROOT, "alphaclip_softmask_method3_robust_0.5_blur"))
    ap.add_argument("--patchcore_src", default=DEFAULT_PATCHCORE, help="patchcore-inspection/src 경로")
    ap.add_argument("--categories", nargs="*", default=None, help="기본: dataset_root 의 모든 카테고리")
    ap.add_argument("--single", action="store_true", help="WideResNet50 단일 모델 (ensemble 대신)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="coreset 샘플링 등 난수 고정")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if not os.path.isdir(args.patchcore_src):
        sys.exit(f"patchcore-inspection 을 찾을 수 없습니다: {args.patchcore_src}\n"
                 "  git clone https://github.com/amazon-science/patchcore-inspection external/patchcore-inspection")
    sys.path.insert(0, args.patchcore_src)
    import patchcore.backbones, patchcore.common, patchcore.patchcore, patchcore.sampler  # noqa: E401,F401
    import patchcore as pc_mod

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    categories = args.categories or sorted(d for d in os.listdir(args.dataset_root)
                                           if os.path.isdir(os.path.join(args.dataset_root, d)))
    print(f"device={device}  categories={categories}  mode={'single' if args.single else 'ensemble'}")
    for c in categories:
        print(f"\n[{c}]")
        process_category(pc_mod, c, args, device)
    print("\ndone.")


if __name__ == "__main__":
    main()
