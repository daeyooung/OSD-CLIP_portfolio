# coding=utf-8
"""MVTec AD open-set 분할 데이터셋.

mvtec_ac_pkl/<category>_seed0_unknown_<defect>.pkl 의 dict["df"] (columns: image_path, failureType, trainTestLabel)
를 읽어 Dassl DatasetBase 로 만든다.
  - Training split 의 failureType 들이 known 클래스 (few-shot: 클래스당 FEW_SHOT_K 장, seed 종속 랜덤 선택)
  - Test split 중 known 클래스는 test, 그 외(unknown 결함) 는 open_others
image_path 는 저장소 루트 기준 상대 경로이며 현재 작업 디렉토리 기준으로 해석한다.
"""
import os
import pickle
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class MVTecPKLOpenSet(DatasetBase):

    def __init__(self, cfg):
        pkl_path = cfg.DATASET.PKL_PATH
        if not pkl_path:
            raise ValueError("cfg.DATASET.PKL_PATH is empty")
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"PKL not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)["df"]

        missing = {"image_path", "failureType", "trainTestLabel"} - set(df.columns)
        if missing:
            raise KeyError(f"missing df columns: {sorted(missing)}")

        split = df["trainTestLabel"].astype(str)
        train_df = df[split == "Training"].reset_index(drop=True)
        test_df = df[split == "Test"].reset_index(drop=True)

        classnames = sorted(train_df["failureType"].astype(str).unique().tolist())
        if not classnames:
            raise ValueError("No known classes in Training split")
        class_to_label = {c: i for i, c in enumerate(classnames)}
        print(f"[MVTecPKLOpenSet] known classes: {classnames}")

        root = os.getcwd()

        def to_datum(row, label):
            impath = os.path.normpath(os.path.join(root, str(row["image_path"])))
            return Datum(impath=impath, label=label, classname=str(row["failureType"]))

        train_x = [to_datum(r, class_to_label[str(r["failureType"])]) for _, r in train_df.iterrows()
                   if str(r["failureType"]) in class_to_label]

        test, open_others = [], []
        for _, r in test_df.iterrows():
            cname = str(r["failureType"])
            if cname in class_to_label:
                test.append(to_datum(r, class_to_label[cname]))
            else:
                open_others.append(to_datum(r, 0))

        # few-shot: 클래스별로 경로 정렬 후 random.Random(f"{seed}_{classname}") 로 K 장 선택
        #   (클래스별 독립 rng 라 순회 순서와 무관하게 재현되고, 같은 pkl + seed 면 항상 같은 샘플)
        seed = int(cfg.SEED) if cfg.SEED is not None and cfg.SEED >= 0 else 0
        k = int(cfg.DATASET.FEW_SHOT_K)
        if k > 0:
            by_class = defaultdict(list)
            for item in train_x:
                by_class[item.classname].append(item)
            selected = []
            for cname in sorted(by_class):
                items = sorted(by_class[cname], key=lambda d: d.impath)
                if len(items) > k:
                    items = sorted(random.Random(f"{seed}_{cname}").sample(items, k), key=lambda d: d.impath)
                selected.extend(items)
            print(f"[MVTecPKLOpenSet] few-shot K={k}: train {len(train_x)} -> {len(selected)} (seed={seed})")
            train_x = selected

        print(f"[MVTecPKLOpenSet] train {len(train_x)} / test(known) {len(test)} / open_others(unknown) {len(open_others)}")
        super().__init__(train_x=train_x, val=[], test=test, open_others=open_others)
        self._classnames = classnames
        self._lab2cname = {i: c for i, c in enumerate(classnames)}
        self._cname2lab = dict(class_to_label)
