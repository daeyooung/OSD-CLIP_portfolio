# coding=utf-8
"""실행 전 환경 점검: 패키지, GPU, 데이터셋 / softmask / Alpha-CLIP 체크포인트 / pkl 파일 존재 여부.

    python check_setup.py
"""
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
              "pill", "screw", "tile", "transistor", "wood", "zipper"]
PACKAGES = ["torch", "torchvision", "numpy", "pandas", "sklearn", "cv2", "PIL", "yacs", "tqdm",
            "matplotlib", "seaborn", "dassl", "alpha_clip"]


def check_packages():
    print("== packages ==")
    ok = True
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]      {name:12s} {getattr(mod, '__version__', '')}")
        except ImportError:
            hint = {"dassl": "pip install -e Dassl.pytorch",
                    "alpha_clip": "README '환경 설정' 의 Alpha-CLIP 설치 참고"}.get(name, "pip install -r requirements.txt")
            print(f"  [MISSING] {name:12s} -> {hint}")
            ok = False
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import clip  # noqa: F401  (저장소 내 clip/ 패키지)
        print("  [ok]      clip (local package)")
    except Exception as e:  # pragma: no cover
        print(f"  [ERROR]   clip (local package): {e}")
        ok = False
    return ok


def check_gpu():
    print("== gpu ==")
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        print("  [WARN] CUDA 를 사용할 수 없습니다. 학습은 GPU 를 전제로 합니다.")
        return False
    for i in range(torch.cuda.device_count()):
        print(f"  [ok] cuda:{i} {torch.cuda.get_device_name(i)}")
    return True


def check_dataset():
    print("== dataset: Dataset/mvtec_ac ==")
    ok = True
    root = REPO_ROOT / "Dataset" / "mvtec_ac"
    if not root.is_dir():
        print(f"  [MISSING] {root}  (README 'Data' 참고)")
        return False
    for c in CATEGORIES:
        good = root / c / "test" / "good"
        if good.is_dir():
            n = sum(1 for _ in (root / c).rglob("*.png"))
            print(f"  [ok] {c:11s} {n:5d} png")
        else:
            print(f"  [MISSING] {good}")
            ok = False
    return ok


def check_softmask():
    print("== softmask: alphaclip_softmask_method3_robust_0.5_blur ==")
    ok = True
    root = REPO_ROOT / "alphaclip_softmask_method3_robust_0.5_blur"
    if not root.is_dir():
        print(f"  [MISSING] {root}  (README 'Data' 참고)")
        return False
    for c in CATEGORIES:
        key = "metal" if c == "metal_nut" else c
        d = root / f"alphaclip_masks_method3_robust_blur_ac_{key}"
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*.npy"))
            print(f"  [ok] {c:11s} {n:5d} npy")
        else:
            print(f"  [MISSING] {d}")
            ok = False
    return ok


def check_files():
    print("== checkpoint / pkl ==")
    ok = True
    ckpt = Path(os.environ.get("ALPHA_CLIP_CKPT", REPO_ROOT / "checkpoints" / "clip_l14_grit20m_fultune_2xe.pth"))
    if ckpt.is_file():
        print(f"  [ok] Alpha-CLIP ckpt {ckpt} ({ckpt.stat().st_size / 2**20:.0f} MB)")
    else:
        print(f"  [MISSING] Alpha-CLIP ckpt {ckpt}  (README '환경 설정' 참고)")
        ok = False
    pkls = sorted((REPO_ROOT / "mvtec_ac_pkl").glob("*.pkl"))
    print(f"  [{'ok' if len(pkls) == 64 else 'WARN'}] mvtec_ac_pkl: {len(pkls)} files (expected 64)")
    return ok


def main():
    print(f"repo root: {REPO_ROOT}\npython   : {sys.version.split()[0]}")
    results = [check_packages(), check_gpu(), check_dataset(), check_softmask(), check_files()]
    print("\nALL OK" if all(results) else "\nSOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
