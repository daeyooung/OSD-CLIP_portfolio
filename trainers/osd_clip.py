# coding=utf-8
"""OSD-CLIP: anomaly-aware local prompt learning for open-set industrial defect recognition.

구성
  - Alpha-CLIP ViT-L/14 (frozen). 결함 영역 이미지는 softmask 를 alpha 채널로, 정상 영역은 alpha = 0 으로 인코딩
  - 패치 특징: CLIP-AD 방식 V-V attention surgery path 를 block 6/12/18/24 에서 추출해 4-stage cosine 유사도 평균
  - 학습 파라미터: known prompt 공유 context (L=16) + unknown prompt M 개의 context (M=30)
  - 손실: L_defect (softmask ≥ τd 패치 ↔ known GT, unknown 은 ε label smoothing)
          + λ · L_normal (정상 영역에서 소스 GT class 유사도 top-Q 패치 ↔ unknown 균등 분포)
  - 추론: ρ=0.20 중심 영역 1개, softmask ≥ τd 패치의 per-patch softmax 평균 → known 점수, 최대값 = MSP
  - 체크포인트 선택 없음: 마지막 epoch 프롬프트로 open-set 평가
"""
import os
import os.path as osp
import random
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.evaluation import compute_overall_metrics, compute_oscr, metric_ood
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, mkdir_if_missing

from clip import clip
import trainers.data_wrapper as dw
from trainers.data_wrapper import RegionCropDataset, sample_normal_region_crop

PATCH_SIZE = {"ViT-L/14": 14, "ViT-B/16": 16}
NUM_NORMAL_VIEWS = 9   # 학습 이미지당 정상 영역 crop 수 (결함 영역 9 view 와 1:1)
TOKEN_BUFFER_KEYS = ("token_prefix", "token_suffix", "neg_token_prefix", "neg_token_suffix")  # 체크포인트에서 복원하지 않는 버퍼


# --------------------------------------------------------------------------- #
# Alpha-CLIP 로드 / 패치 특징 추출
# --------------------------------------------------------------------------- #

def load_alpha_clip(cfg):
    """Alpha-CLIP (이미지 + alpha 채널) 을 CPU 에 로드한다. 체크포인트: env ALPHA_CLIP_CKPT > cfg 값."""
    import alpha_clip
    ckpt = os.environ.get("ALPHA_CLIP_CKPT", cfg.TRAINER.OSDCLIP.ALPHA_CLIP_CKPT)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Alpha-CLIP checkpoint not found: {ckpt}\n"
            "  https://github.com/SunzeY/AlphaCLIP 의 clip_l14_grit20m_fultune_2xe.pth 를 checkpoints/ 에 두거나 "
            "ALPHA_CLIP_CKPT 환경변수로 경로를 지정하세요.")
    print(f"[OSD-CLIP] loading Alpha-CLIP {cfg.MODEL.BACKBONE.NAME} from {ckpt}")
    model, _ = alpha_clip.load(cfg.MODEL.BACKBONE.NAME, alpha_vision_ckpt_pth=ckpt, device="cpu")
    return model


def add_local_feature_forward(visual):
    """Alpha-CLIP VisionTransformer 에 CLIP-AD surgery 방식 4-stage 패치 특징 forward 를 추가한다.

    - 각 block 에서 V-V attention (Q=K=V) 출력을 계산해 surgery path 에 누적 (MLP 없음)
    - n//4 block 마다 surgery path 를 리셋하고, block [n/4, n/2, 3n/4, n] 끝에서 패치 특징을 추출
    - 표준 path 의 CLS 토큰은 global 특징으로 반환
    """

    def vv_attention(block, x):
        y = block.ln_1(x)                                        # [L, N, C]
        _, _, v = block.attn.in_proj(y).chunk(3, dim=-1)
        L, N, C = v.shape
        H = block.attn.num_heads
        v_mh = v.contiguous().view(L, N * H, C // H).transpose(0, 1)          # [N*H, L, hd]
        attn = torch.bmm(v_mh * block.attn.scale, v_mh.transpose(-2, -1)).softmax(dim=-1)
        out = torch.bmm(attn, v_mh).transpose(0, 1).reshape(L, N, C)
        return block.attn.out_proj(out)

    def forward_local_features(self, x, alpha):
        x = self.conv1(x) + self.conv1_alpha(alpha)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)             # [B, grid^2, width]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)                                    # NLD -> LND

        blocks = list(self.transformer.resblocks)
        n = len(blocks)
        target_ids = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        fusion_step = n // 4

        x_ori, x_s, stage_feats = x, None, []
        for i, block in enumerate(blocks):
            vv = vv_attention(block, x_ori)
            x_s = vv if (x_s is None or i % fusion_step == 0) else x_s + vv
            if i in target_ids:
                stage_feats.append(x_s.clone())
            x_ori = block(x_ori)

        global_feat = self.ln_post(x_ori.permute(1, 0, 2)[:, 0, :])
        if self.proj is not None:
            global_feat = global_feat @ self.proj

        local_feats = []
        for v_raw in stage_feats:
            lf = self.ln_post(v_raw.permute(1, 0, 2))[:, 1:, :]               # CLS 제거 -> [B, N_patches, C]
            if self.proj is not None:
                lf = lf @ self.proj
            local_feats.append(lf)
        return global_feat, local_feats

    visual.forward_local_features = types.MethodType(forward_local_features, visual)


# --------------------------------------------------------------------------- #
# 텍스트 인코더 / 프롬프트
# --------------------------------------------------------------------------- #

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


class PromptLearner(nn.Module):
    """known prompt : ctx(L) + "{object} {defect class}."   (ctx 공유)
       unknown prompt: neg_ctx_m(L) + "{object}."           (m = 1..M, 각자 context)
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        mc = cfg.TRAINER.OSDCLIP
        n_ctx, n_neg = mc.N_CTX, mc.NUM_NEG_PROMPTS
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        object_name = cfg.DATASET.OBJECT_NAME.replace("_", " ").strip()
        if not object_name:
            raise ValueError("DATASET.OBJECT_NAME must be set (e.g. 'metal nut')")

        self.n_cls = len(classnames)
        self.num_neg_prompts = n_neg
        placeholder = " ".join(["X"] * n_ctx)

        # known
        ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)
        names = [c.replace("_", " ") for c in classnames]
        pfx = object_name.lower() + " "
        names = [c[len(object_name):].strip() if c.lower().startswith(pfx) else c for c in names]
        tokenized = torch.cat([clip.tokenize(f"{placeholder} {object_name} {c}.") for c in names])
        with torch.no_grad():
            emb = clip_model.token_embedding(tokenized).type(dtype)
        self.register_buffer("token_prefix", emb[:, :1, :])
        self.register_buffer("token_suffix", emb[:, 1 + n_ctx:, :])
        self.tokenized_prompts = tokenized

        # unknown
        neg_ctx = torch.empty(n_neg, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(neg_ctx, std=0.02)
        self.neg_ctx = nn.Parameter(neg_ctx)
        neg_tokenized = torch.cat([clip.tokenize(f"{placeholder} {object_name}.") for _ in range(n_neg)])
        with torch.no_grad():
            neg_emb = clip_model.token_embedding(neg_tokenized).type(dtype)
        self.register_buffer("neg_token_prefix", neg_emb[:, :1, :])
        self.register_buffer("neg_token_suffix", neg_emb[:, 1 + n_ctx:, :])
        self.neg_tokenized_prompts = neg_tokenized

        print(f"[OSD-CLIP] known prompt  : '{placeholder} {object_name} {{class}}.' x {self.n_cls}")
        print(f"[OSD-CLIP] unknown prompt: '{placeholder} {object_name}.' x {n_neg}")

    def forward(self):
        ctx_k = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        known = torch.cat([self.token_prefix, ctx_k, self.token_suffix], dim=1)
        unknown = torch.cat([self.neg_token_prefix, self.neg_ctx, self.neg_token_suffix], dim=1)
        return known, unknown


class OSDCLIPModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_known_classes = len(classnames)

    def get_text_features(self):
        """L2 정규화된 (known [n_cls, D], unknown [M, D])"""
        known, unknown = self.prompt_learner()
        k = self.text_encoder(known, self.prompt_learner.tokenized_prompts)
        u = self.text_encoder(unknown, self.prompt_learner.neg_tokenized_prompts)
        return F.normalize(k.float(), dim=-1), F.normalize(u.float(), dim=-1)

    def encode_local(self, image, alpha):
        """4-stage 패치 특징, 각 [B, N_patches, D] (L2 정규화)"""
        _, local_feats = self.image_encoder.forward_local_features(image.type(self.dtype), alpha.type(self.dtype))
        return [F.normalize(f.float(), dim=-1) for f in local_feats]

    def encode_global(self, image, alpha):
        return F.normalize(self.image_encoder(image.type(self.dtype), alpha.type(self.dtype)).float(), dim=-1)


def fuse_logits(local_feats, text_feats, logit_scale):
    """4-stage cosine 유사도 평균 x logit_scale -> [B, N, n_text]"""
    return logit_scale * sum(lf @ text_feats.t() for lf in local_feats) / len(local_feats)


# --------------------------------------------------------------------------- #
# 손실
# --------------------------------------------------------------------------- #

def loss_defect_regions(known_logits, unknown_logits, softmask_patches, label,
                        threshold, top_k, T, ols, n_cls, n_neg):
    """결함 영역: softmask ≥ threshold 인 패치(없으면 softmask 상위 top_k)마다
    known GT = 1, 다른 known = 0, unknown = ols 인 soft target 으로 CE."""
    losses = []
    for b in range(known_logits.shape[0]):
        sm = softmask_patches[b]
        sel = (sm >= threshold).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            sel = sm.topk(k=min(top_k, sm.shape[0]), dim=0)[1]
        logits = torch.cat([known_logits[b][sel], unknown_logits[b][sel]], dim=1) / T   # [k, n_cls+n_neg]
        target = torch.zeros(n_cls + n_neg, device=logits.device)
        target[label[b]] = 1.0
        target[n_cls:] = ols
        losses.append(F.cross_entropy(logits, target.unsqueeze(0).expand(logits.shape[0], -1)))
    return torch.stack(losses).mean()


def loss_normal_regions(unknown_logits, known_logits, source_label, top_k, T, n_cls, n_neg):
    """정상 영역: 소스 이미지 GT known class 유사도 상위 top_k 패치를 unknown 균등 분포(1/M) 로 CE."""
    B, N, _ = unknown_logits.shape
    k = min(top_k, N)
    label_idx = source_label.view(B, 1, 1).expand(B, N, 1)
    topk_idx = known_logits.gather(2, label_idx).topk(k=k, dim=1)[1]                   # [B, k, 1]
    k_logits = known_logits.gather(1, topk_idx.expand(-1, -1, n_cls))
    u_logits = unknown_logits.gather(1, topk_idx.expand(-1, -1, n_neg))
    logits = torch.cat([k_logits, u_logits], dim=2) / T
    target = torch.zeros(B, n_cls + n_neg, device=logits.device)
    target[:, n_cls:] = 1.0 / n_neg
    target = target.unsqueeze(1).expand(-1, k, -1)
    return F.cross_entropy(logits.reshape(-1, n_cls + n_neg), target.reshape(-1, n_cls + n_neg))


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

@TRAINER_REGISTRY.register()
class OSDCLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.MODEL.BACKBONE.NAME in PATCH_SIZE, f"unsupported backbone {cfg.MODEL.BACKBONE.NAME}"

    # ── data ──────────────────────────────────────────────────────────────────
    def build_data_loader(self):
        cfg = self.cfg
        dm = DataManager(cfg, dataset_wrapper=RegionCropDataset)   # test / open_others 로더

        train_set = RegionCropDataset(cfg, dm.dataset.train_x, is_train=True)   # 이미지 x 9 view
        seed = int(cfg.SEED) if cfg.SEED >= 0 else 0
        g = torch.Generator()
        g.manual_seed(seed)

        def seed_worker(_):
            s = torch.initial_seed() % 2 ** 32
            np.random.seed(s)
            random.seed(s)

        self.train_loader_x = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            sampler=torch.utils.data.RandomSampler(train_set, generator=g),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
            worker_init_fn=seed_worker,
        )
        self.train_loader_u = None
        self.val_loader = None
        self.test_loader = dm.test_loader
        self.others_loader = dm.others_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm
        print(f"[OSD-CLIP] train images {len(dm.dataset.train_x)} (x{train_set.K} views) | "
              f"test(known) {len(dm.dataset.test)} | open_others(unknown) {len(dm.dataset.open_others)}")

    # ── model ─────────────────────────────────────────────────────────────────
    def build_model(self):
        cfg = self.cfg
        mc = cfg.TRAINER.OSDCLIP
        dw.SOFTMASK_DIR = mc.SOFTMASK_DIR
        print(f"[OSD-CLIP] SOFTMASK_DIR = {mc.SOFTMASK_DIR}")

        clip_model = load_alpha_clip(cfg)
        add_local_feature_forward(clip_model.visual)
        clip_model.float()

        classnames = self.dm.dataset.classnames
        self.model = OSDCLIPModel(cfg, classnames, clip_model)
        for name, p in self.model.named_parameters():
            if "prompt_learner" not in name:
                p.requires_grad_(False)
        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.lam = mc.LAMBDA
        self.top_k = mc.TOPK
        self.T = mc.T
        self.softmask_threshold = mc.SOFTMASK_THRESHOLD
        self.ols = mc.OLS
        self.crop_size = cfg.INPUT.SIZE[0]
        self.n_patches_side = self.crop_size // PATCH_SIZE[cfg.MODEL.BACKBONE.NAME]
        print(f"[OSD-CLIP] n_cls={len(classnames)} M={mc.NUM_NEG_PROMPTS} L={mc.N_CTX} lambda={self.lam} "
              f"Q={self.top_k} tau_d={self.softmask_threshold} eps={self.ols} T={self.T} "
              f"patches={self.n_patches_side ** 2}")

        # 정상 영역 crop 풀: 학습 이미지마다 9개, run 시작 시 1회 생성 후 고정
        to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
        items = self.dm.dataset.train_x * NUM_NORMAL_VIEWS
        self.normal_crops = [to_tensor(sample_normal_region_crop(it.impath, self.crop_size, mc.NORMAL_CROP_ROI_SIZE))
                             for it in items]
        self.normal_labels = [it.label for it in items]
        self._pool_queue, self._pool_epoch = [], -1
        print(f"[OSD-CLIP] normal-region crops: {len(self.normal_crops)} (images x {NUM_NORMAL_VIEWS}, "
              f"roi={mc.NORMAL_CROP_ROI_SIZE}px)")

    def _shuffled_pool(self):
        # 원본 실험 코드와 같은 난수 소비(list + random.shuffle). random.sample 로 바꾸면 같은 seed 에서도 순서가 달라진다.
        idx = list(range(len(self.normal_crops)))
        random.shuffle(idx)
        return idx

    def _sample_normal_crops(self, B):
        """epoch 마다 풀을 셔플해 순서대로 B 개씩 꺼낸다 (소스 이미지 GT label 함께 반환)."""
        if self._pool_epoch != self.epoch:
            self._pool_queue = self._shuffled_pool()
            self._pool_epoch = self.epoch
        picked = []
        while len(picked) < B:
            need = B - len(picked)
            picked += self._pool_queue[:need]
            self._pool_queue = self._pool_queue[need:]
            if not self._pool_queue:
                self._pool_queue = self._shuffled_pool()
        crops = torch.stack([self.normal_crops[i] for i in picked]).to(self.device)
        labels = torch.tensor([self.normal_labels[i] for i in picked], dtype=torch.long, device=self.device)
        return crops, labels

    # ── batch parsing ─────────────────────────────────────────────────────────
    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device), batch["mask"].to(self.device)

    def parse_batch_test(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device), batch["mask"].to(self.device)

    def _patch_scores(self, mask):
        """softmask [B,1,H,W] -> 패치 단위 평균 [B, N_patches]"""
        return F.adaptive_avg_pool2d(mask.float(), (self.n_patches_side, self.n_patches_side)).flatten(1)

    def _save_visuals(self, image, mask, normal, subdir_tag):
        mean = torch.tensor(self.cfg.INPUT.PIXEL_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.cfg.INPUT.PIXEL_STD).view(1, 3, 1, 1)
        out = osp.join(self.output_dir, "input_samples")
        mkdir_if_missing(out)
        save_image(torch.clamp(image.cpu() * std + mean, 0, 1), osp.join(out, f"{subdir_tag}_defect_regions.jpg"), nrow=4)
        save_image(mask[:, :1].cpu(), osp.join(out, f"{subdir_tag}_softmask.jpg"), nrow=4)
        if normal is not None:
            save_image(torch.clamp(normal.cpu() * std + mean, 0, 1), osp.join(out, f"{subdir_tag}_normal_regions.jpg"), nrow=4)

    # ── training ──────────────────────────────────────────────────────────────
    def forward_backward(self, batch):
        image, label, mask = self.parse_batch_train(batch)
        B = image.shape[0]
        n_cls = self.model.num_known_classes
        n_neg = self.model.prompt_learner.num_neg_prompts

        normal_image, normal_label = self._sample_normal_crops(B)
        alpha_defect = mask.float()                        # 결함 영역: softmask 를 alpha 로
        alpha_normal = torch.zeros_like(alpha_defect)      # 정상 영역: alpha = 0
        if self.epoch == 0 and self.batch_idx == 0:
            self._save_visuals(image, mask, normal_image, "train_epoch0")

        known_feats, unknown_feats = self.model.get_text_features()
        logit_scale = self.model.logit_scale.exp().clamp(max=100.0)

        defect_locals = self.model.encode_local(image, alpha_defect)
        normal_locals = self.model.encode_local(normal_image, alpha_normal)

        d_known = fuse_logits(defect_locals, known_feats, logit_scale)       # [B, N, n_cls]
        d_unknown = fuse_logits(defect_locals, unknown_feats, logit_scale)   # [B, N, M]
        n_known = fuse_logits(normal_locals, known_feats, logit_scale)
        n_unknown = fuse_logits(normal_locals, unknown_feats, logit_scale)

        l_defect = loss_defect_regions(d_known, d_unknown, self._patch_scores(mask), label,
                                       self.softmask_threshold, self.top_k, self.T, self.ols, n_cls, n_neg)
        l_normal = loss_normal_regions(n_unknown, n_known, normal_label, self.top_k, self.T, n_cls, n_neg)
        loss = l_defect + self.lam * l_normal
        self.model_backward_and_update(loss)

        with torch.no_grad():   # 학습 모니터링용 closed-set 정확도 (global CLS 특징)
            logits_acc = logit_scale * self.model.encode_global(image, alpha_defect) @ known_feats.t()

        summary = {
            "loss": loss.item(),
            "loss_defect": l_defect.item(),
            "loss_normal": l_normal.item(),
            "acc": compute_accuracy(logits_acc, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return summary

    def after_epoch(self):
        """모델 선택 없음: 마지막 epoch (및 CHECKPOINT_FREQ) 에만 저장한다."""
        last = (self.epoch + 1) == self.max_epoch
        freq = self.cfg.TRAIN.CHECKPOINT_FREQ
        if last or (freq > 0 and (self.epoch + 1) % freq == 0):
            self.save_model(self.epoch, self.output_dir)

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def model_inference(self, image, mask):
        """[B, n_cls + M]: known 열 = softmask ≥ τd 패치(없으면 상위 Q)의 per-patch softmax 평균, unknown 열 = 0.
        known 열의 최대값이 MSP(known 신뢰도), argmax 가 예측 클래스."""
        B = image.shape[0]
        n_cls = self.model.num_known_classes
        n_neg = self.model.prompt_learner.num_neg_prompts

        known_feats, unknown_feats = self.model.get_text_features()
        logit_scale = self.model.logit_scale.exp().clamp(max=100.0)
        local_feats = self.model.encode_local(image, mask.float())
        all_logits = torch.cat([fuse_logits(local_feats, known_feats, logit_scale),
                                fuse_logits(local_feats, unknown_feats, logit_scale)], dim=-1)   # [B, N, n_cls+M]
        exp_all = torch.exp((all_logits - all_logits.max(dim=-1, keepdim=True)[0]) / self.T)
        smax_known = exp_all[..., :n_cls] / exp_all.sum(dim=-1, keepdim=True)                   # [B, N, n_cls]

        patch_scores = self._patch_scores(mask)
        per_sample = []
        for b in range(B):
            sel = (patch_scores[b] >= self.softmask_threshold).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                sel = patch_scores[b].topk(k=min(self.top_k, patch_scores.shape[1]), dim=0)[1]
            per_sample.append(smax_known[b][sel].mean(dim=0))
        result = torch.zeros(B, n_cls + n_neg, device=self.device)
        result[:, :n_cls] = torch.stack(per_sample)
        return result

    @torch.no_grad()
    def test(self, split=None):
        """closed-set 정확도 (known test). Dassl after_train 에서 호출."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        for batch in tqdm(self.test_loader, desc="test(known)"):
            image, label, mask = self.parse_batch_test(batch)
            self.evaluator.process(self.model_inference(image, mask), label)
        results = self.evaluator.evaluate()
        for k, v in results.items():
            self.write_scalar(f"test/{k}", v, self.epoch)
        return list(results.values())[0]

    @torch.no_grad()
    def open_test(self):
        """open-set 평가: known test + unknown(open_others). AUROC / OSCR / Acc_Known / Best_F1 등을 CSV 로 저장."""
        self.set_model_mode("eval")
        n_cls = self.num_classes

        def run(loader, tag, with_label):
            scores, labels = [], []
            for i, batch in enumerate(tqdm(loader, desc=tag)):
                image, label, mask = self.parse_batch_test(batch) if with_label else (
                    batch["img"].to(self.device), None, batch["mask"].to(self.device))
                if i == 0:
                    self._save_visuals(image, mask, None, tag)
                scores.append(self.model_inference(image, mask)[:, :n_cls].cpu().numpy())
                if with_label:
                    labels.append(label.cpu().numpy())
            return np.concatenate(scores), (np.concatenate(labels) if with_label else None)

        pred_k, labels = run(self.test_loader, "test_known", True)
        pred_u, _ = run(self.others_loader, "test_unknown", False)

        results = metric_ood(np.max(pred_k, axis=1), np.max(pred_u, axis=1))["Bas"]   # AUROC 등
        pred_cls_k = np.argmax(pred_k, axis=1)
        results["Acc_Known"] = np.mean(pred_cls_k == labels) * 100.0
        results["OSCR"] = compute_oscr(pred_k, pred_u, labels) * 100.0
        best_acc, best_f1, best_th, unk_recall = compute_overall_metrics(pred_k, pred_u, labels, n_cls)
        results.update(Best_Acc=best_acc * 100.0, Best_F1=best_f1 * 100.0, Best_Th=best_th,
                       Unknown_Recall=unk_recall * 100.0)

        # best threshold 에서의 known recall 과 confusion matrix
        y_true = np.concatenate([labels, np.full(len(pred_u), n_cls)])
        y_pred = np.concatenate([pred_cls_k, np.argmax(pred_u, axis=1)])
        y_pred[np.concatenate([pred_k.max(1), pred_u.max(1)]) < best_th] = n_cls
        is_known = y_true != n_cls
        results["Known_Recall"] = (np.mean(y_pred[is_known] == y_true[is_known]) * 100.0) if is_known.any() else 0.0

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted (last = unknown)")
        plt.ylabel("True (last = unknown)")
        plt.title(f"Confusion matrix (threshold {best_th:.4f})")
        plt.savefig(osp.join(self.cfg.OUTPUT_DIR, "confusion_matrix_open_others_msp.png"))
        plt.close()

        print(" AUROC {:.3f} | OSCR {:.3f} | Acc_Known {:.3f} | Best_F1 {:.3f} | Unknown_Recall {:.3f} | Known_Recall {:.3f}"
              .format(results["AUROC"], results["OSCR"], results["Acc_Known"], results["Best_F1"],
                      results["Unknown_Recall"], results["Known_Recall"]))

        df = pd.DataFrame([results])
        df["MSP"] = True
        df["Open_Split"] = "open_others"
        out_csv = osp.join(self.cfg.OUTPUT_DIR, "open_test_results_open_others_msp.csv")
        df.to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")
        return results

    # ── checkpoint ────────────────────────────────────────────────────────────
    def load_model(self, directory, epoch=None):
        """model.pth.tar-<epoch> 를 로드한다 (기본: 마지막 epoch = OPTIM.MAX_EPOCH)."""
        if not directory:
            raise ValueError("--model-dir is required")
        epoch = int(epoch) if epoch is not None else int(self.cfg.OPTIM.MAX_EPOCH)
        for name in self.get_model_names():
            path = osp.join(directory, name, f"model.pth.tar-{epoch}")
            if not osp.exists(path):
                raise FileNotFoundError(f"checkpoint not found: {path}")
            ckpt = load_checkpoint(path)
            state = ckpt["state_dict"]
            for key in TOKEN_BUFFER_KEYS:
                state.pop(key, None)        # 토큰 임베딩 버퍼는 클래스 이름으로 다시 계산
            print(f'[OSD-CLIP] loading {name} from "{path}" (epoch {ckpt["epoch"]})')
            result = self._models[name].load_state_dict(state, strict=False)
            missing = [k for k in result.missing_keys if k not in TOKEN_BUFFER_KEYS]
            if missing or result.unexpected_keys:
                raise RuntimeError(f"checkpoint key mismatch: missing={missing} unexpected={list(result.unexpected_keys)}")
