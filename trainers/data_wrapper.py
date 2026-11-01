# coding=utf-8
"""PatchCore softmask 기반 영역(region) 생성과 데이터셋 래퍼.

- load_softmask()               : 이미지 경로 -> softmask(.npy, 값 범위 [0, 1]) 로드
- sample_fixed_ratio_crops()    : softmask 의 top-1 component 주변에서 결함 영역 비율이 ρ 가 되도록 crop
                                  (ρ = 0.10 : 상하좌우 shift 4개, ρ = 0.20 : 중심 + shift 4개 -> 학습용 9 view)
- sample_normal_region_crop()   : softmask ≤ 0.5 인 정상 영역에서 roi_size 정사각 crop (unknown prompt 학습용)
- RegionCropDataset             : 학습 시 이미지당 9 view, 평가 시 ρ = 0.20 중심 view 1개를 반환하는 Dataset
"""
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image

# 카테고리별 softmask 루트. 트레이너가 cfg.TRAINER.OSDCLIP.SOFTMASK_DIR 값으로 덮어쓴다.
#   <SOFTMASK_DIR>/<defect_class>/softmask/<image_stem>.npy
SOFTMASK_DIR = "./softmask"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------- #
# softmask
# --------------------------------------------------------------------------- #

def softmask_path(impath):
    class_name = os.path.basename(os.path.dirname(impath))
    stem = os.path.splitext(os.path.basename(impath))[0]
    return os.path.join(SOFTMASK_DIR, class_name, "softmask", f"{stem}.npy")


def load_softmask(impath, H, W):
    """float32 [H, W] softmask. good(정상) 이미지는 파일이 없으면 전체 1 로 채운다."""
    path = softmask_path(impath)
    if os.path.isfile(path):
        sm = np.load(path).astype(np.float32)
        if sm.shape != (H, W):
            sm = cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
        return sm
    if os.path.basename(os.path.dirname(impath)) == "good":
        return np.ones((H, W), dtype=np.float32)
    raise FileNotFoundError(f"[softmask missing] image: {impath}\n  expected: {path}")


# --------------------------------------------------------------------------- #
# top-1 component (전역 최대값 픽셀 기준) 과 weighted bbox
# --------------------------------------------------------------------------- #

def select_top_component_by_peak(soft_mask, threshold_ratio=0.5, min_area=5):
    """softmask 전역 최대값 픽셀을 포함하는 connected component 를 top-1 으로 선택한다.

    1) max_val * threshold_ratio 로 이진화해 connected components 를 구한다.
    2) 최대값 픽셀을 포함하는 component 만 후보로 둔다.
    3) 후보가 여럿이면 component 내 softmask 평균이 가장 큰 것을 고른다.
    """
    mask = soft_mask.astype(np.float32).copy()
    H, W = mask.shape
    max_val = float(mask.max())
    if max_val <= 1e-8:
        return np.zeros((H, W), dtype=np.float32)

    peak_yx = np.argwhere(mask >= max_val - 1e-6)
    binary = (mask >= max_val * threshold_ratio).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = {int(labels[y, x]) for y, x in peak_yx if labels[y, x] > 0}
    if not candidates:                       # 최대값 픽셀이 이진화에서 빠진 경우: 그 픽셀만 사용
        top_mask = np.zeros((H, W), dtype=np.float32)
        py, px = int(peak_yx[0][0]), int(peak_yx[0][1])
        top_mask[py, px] = mask[py, px]
        return top_mask

    if len(candidates) == 1:
        best_label = next(iter(candidates))
    else:
        best_label, best_score = None, -1.0
        for lbl in candidates:
            if int(stats[lbl, cv2.CC_STAT_AREA]) < min_area:
                continue
            comp_values = mask[labels == lbl]
            if comp_values.size == 0:
                continue
            score = float(comp_values.mean())
            if score > best_score:
                best_score, best_label = score, lbl
        if best_label is None:
            best_label = next(iter(candidates))

    top_mask = np.zeros((H, W), dtype=np.float32)
    top_mask[labels == best_label] = mask[labels == best_label]
    return top_mask


def top_component_weighted_stats(soft_mask, threshold_ratio=0.5, std_scale=3.0, min_area=5):
    """top-1 component 안에서 softmask 가중 중심 (cx, cy) 과 bbox (bw, bh) 를 추정한다."""
    mask = soft_mask.astype(np.float32).copy()
    H, W = mask.shape
    top_mask = select_top_component_by_peak(mask, threshold_ratio=threshold_ratio, min_area=min_area)

    total = float(top_mask.sum())
    if total <= 1e-8:
        cx, cy = W / 2.0, H / 2.0
        bw, bh = max(2, W // 3), max(2, H // 3)
        return cx, cy, bw, bh, int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)

    ys, xs = np.indices(top_mask.shape)
    cx = float((xs * top_mask).sum() / total)
    cy = float((ys * top_mask).sum() / total)
    var_x = float((((xs - cx) ** 2) * top_mask).sum() / total)
    var_y = float((((ys - cy) ** 2) * top_mask).sum() / total)
    std_x = np.sqrt(max(var_x, 1e-8))
    std_y = np.sqrt(max(var_y, 1e-8))

    bw = max(2, int(np.ceil(std_scale * std_x)))
    bh = max(2, int(np.ceil(std_scale * std_y)))
    x_min = clamp(int(round(cx - bw / 2.0)), 0, W - 2)
    x_max = clamp(int(round(cx + bw / 2.0)), 1, W - 1)
    y_min = clamp(int(round(cy - bh / 2.0)), 0, H - 2)
    y_max = clamp(int(round(cy + bh / 2.0)), 1, H - 1)
    return cx, cy, bw, bh, x_min, y_min, x_max, y_max


# --------------------------------------------------------------------------- #
# 결함 영역 crop (fixed-ratio, 11 view)
# --------------------------------------------------------------------------- #

# center, left, right, up, down
SHIFTS = [(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]


def sample_fixed_ratio_crops(img, soft_mask, C=224, target_ratios=(0.10, 0.20), threshold_ratio=0.5):
    """결함 영역(top-1 component bbox) 이 crop 안에서 비율 ρ 를 차지하도록 ROI 를 잡아 C×C 로 resize 한다.

    반환: list of (img_crop, mask_crop), 총 1 + 5 * len(target_ratios) 개
      [0]      원본 resize
      [1..5]   ρ = target_ratios[0] : center / left / right / up / down
      [6..10]  ρ = target_ratios[1] : center / left / right / up / down
    ROI: roi_w = bw / sqrt(ρ), roi_h = bh / sqrt(ρ)  ->  (bw / roi_w) * (bh / roi_h) = ρ
    """
    H, W = img.shape[:2]
    sm = soft_mask.astype(np.float32)
    if sm.max() > 1.0:
        sm = sm / 255.0
    cx, cy, bw, bh, x_min, y_min, x_max, y_max = top_component_weighted_stats(sm, threshold_ratio)
    bw, bh = max(bw, 1), max(bh, 1)

    crops = [(cv2.resize(img, (C, C), interpolation=cv2.INTER_AREA),
              cv2.resize(sm, (C, C), interpolation=cv2.INTER_LINEAR).astype(np.float32))]

    for r in target_ratios:
        roi_w = max(bw, int(np.ceil(bw / np.sqrt(r))))
        roi_h = max(bh, int(np.ceil(bh / np.sqrt(r))))
        for sx, sy in SHIFTS:
            ccx = cx + sx * (roi_w - bw) / 2.0
            ccy = cy + sy * (roi_h - bh) / 2.0
            x0 = int(round(ccx - roi_w / 2.0))
            y0 = int(round(ccy - roi_h / 2.0))
            x1, y1 = x0 + roi_w, y0 + roi_h

            # 이미지 경계를 넘으면 ROI 를 안쪽으로 민다
            if x0 < 0:
                x1 -= x0
                x0 = 0
            if y0 < 0:
                y1 -= y0
                y0 = 0
            if x1 > W:
                x0 -= x1 - W
                x1 = W
            if y1 > H:
                y0 -= y1 - H
                y1 = H
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if (x1 - x0) < 2 or (y1 - y0) < 2:
                x0 = clamp(x_min - 1, 0, W - 2)
                y0 = clamp(y_min - 1, 0, H - 2)
                x1 = clamp(x_max + 2, 2, W)
                y1 = clamp(y_max + 2, 2, H)

            img_c = cv2.resize(img[y0:y1, x0:x1], (C, C), interpolation=cv2.INTER_AREA)
            m_c = cv2.resize(sm[y0:y1, x0:x1], (C, C), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            crops.append((img_c, m_c))
    return crops


# --------------------------------------------------------------------------- #
# 정상 영역 crop (unknown prompt 학습용)
# --------------------------------------------------------------------------- #

def sample_normal_region_crop(impath, crop_size, roi_size, max_retries=50):
    """학습 결함 이미지에서 softmask ≤ 0.5 인 픽셀만 포함하는 roi_size 정사각 영역을 랜덤으로 찾아
    crop_size 로 resize 한다. max_retries 안에 못 찾으면 softmask 최댓값이 가장 낮았던 후보를 쓴다."""
    img_pil = read_image(impath)
    W0, H0 = img_pil.size
    img_np = np.array(img_pil)
    softmask = load_softmask(impath, H0, W0)

    pad_h, pad_w = max(0, roi_size - H0), max(0, roi_size - W0)
    if pad_h > 0 or pad_w > 0:
        img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        softmask = np.pad(softmask, ((0, pad_h), (0, pad_w)), mode="reflect")
    H, W = img_np.shape[:2]

    best_max, best_crop = float("inf"), None
    for _ in range(max_retries):
        y0 = random.randint(0, H - roi_size)
        x0 = random.randint(0, W - roi_size)
        m = float(softmask[y0:y0 + roi_size, x0:x0 + roi_size].max())
        if m < best_max:
            best_max = m
            best_crop = img_np[y0:y0 + roi_size, x0:x0 + roi_size].copy()
        if m <= 0.5:
            break
    return Image.fromarray(cv2.resize(best_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class RegionCropDataset(TorchDataset):
    """softmask 기반 결함 영역 crop 데이터셋.

    학습  : 이미지당 K = 9 view (ρ=0.10 left/right/up/down + ρ=0.20 center/left/right/up/down)
    평가  : ρ = 0.20 center view 1개
    반환 dict: img [3, C, C] (CLIP 정규화), mask [1, C, C] (softmask, Alpha-CLIP alpha 채널), label, impath, index
    """
    TRAIN_VIEW_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]   # sample_fixed_ratio_crops 반환 인덱스
    TEST_VIEW_INDEX = 6                                  # ρ = 0.20 center
    TARGET_RATIOS = (0.10, 0.20)

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        # transform 인자는 Dassl DataManager 인터페이스 호환용 (crop/정규화는 내부에서 수행)
        self.data_source = data_source
        self.is_train = is_train
        self.K = len(self.TRAIN_VIEW_INDICES)
        self.crop_size = cfg.INPUT.SIZE[0]
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])

    def __len__(self):
        return len(self.data_source) * self.K if self.is_train else len(self.data_source)

    def __getitem__(self, idx):
        if self.is_train:
            item = self.data_source[idx // self.K]
            view = self.TRAIN_VIEW_INDICES[idx % self.K]
        else:
            item = self.data_source[idx]
            view = self.TEST_VIEW_INDEX

        img_np = np.array(read_image(item.impath))
        H0, W0 = img_np.shape[:2]
        softmask = load_softmask(item.impath, H0, W0)
        img_c, m_c = sample_fixed_ratio_crops(img_np, softmask, C=self.crop_size,
                                              target_ratios=self.TARGET_RATIOS)[view]
        return {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx,
            "img": self.to_tensor(Image.fromarray(img_c)),
            "mask": torch.from_numpy(m_c).unsqueeze(0),
        }
