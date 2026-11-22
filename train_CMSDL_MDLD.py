import argparse
import os
import time
import random
import json
import math
import csv

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import cv2
from tqdm import tqdm
from sklearn import metrics
import scipy.io as io

from utils.data import Data_Loader, to_normalization
from utils.loss import *
from utils.Evaluation import Evaluation

from model.MDLD import (
    DINOStudentTeacher,
    dino_loss,
    weighted_dino_loss,
    layer_token_ce_loss,
)


# ======================
# 小工具
# ======================
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def minmax01(arr: np.ndarray):
    arr = arr.astype(np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax - amin < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)

def to_uint8_01(arr01: np.ndarray):
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0).astype(np.uint8)

class Timer:
    def __init__(self):
        self.t0 = None
    def start(self):
        self.t0 = time.time()
    def stop(self):
        return 0.0 if self.t0 is None else (time.time() - self.t0)


# ======================
# 随机种子
# ======================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ======================
# 数据增强
# ======================
def build_augment_pipelines():
    global_aug1 = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    global_aug2 = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    return global_aug1, global_aug2

def apply_batch_aug(batch: torch.Tensor, aug) -> torch.Tensor:
    return torch.stack([aug(img) for img in batch], dim=0)


# ======================
# Fixed-final objective weighting
# ======================
def get_fixed_final_weights(base_w_sm: float, base_w_cm: float, base_w_token: float):
    """Return constant objective weights used in every training epoch."""
    return float(base_w_sm), float(base_w_cm), float(base_w_token)


# ======================
# 训练循环：CMSDL-MSDL
# ======================
def train_one_epoch(
    model,
    loader,
    opt,
    device,
    epoch: int,
    total_epochs: int,
    t_teacher: float,
    t_student: float,
    dino_w_cm: float = 1.0,
    dino_w_sm: float = 0.5,
    dino_w_token: float = 0.1,
    center_m: float = 0.9,
    teacher_m: float = 0.996,
    student_chunk_size: int = 128,
    conf_power: float = 1.0,
    token_layer_weights=None,
):
    model.train()
    global_aug1, global_aug2 = build_augment_pipelines()

    # Fixed-final: all objective weights are applied unchanged from Epoch 1.
    w_sm_epoch, w_cm_epoch, w_token_epoch = get_fixed_final_weights(
        base_w_sm=dino_w_sm,
        base_w_cm=dino_w_cm,
        base_w_token=dino_w_token,
    )

    running = 0.0
    running_cm = 0.0
    running_sm = 0.0
    running_tok = 0.0
    running_weighted_cm = 0.0
    running_weighted_sm = 0.0
    running_weighted_tok = 0.0
    running_conf = 0.0

    with tqdm(total=len(loader), desc=f"Train (CMSDL-MSDL) [Epoch {epoch}]", ncols=180) as t:
        for batch_idx, (t1, t2, _) in enumerate(loader):
            t1 = t1.float()
            t2 = t2.float()

            # ===== 增强 =====
            t1_v1 = apply_batch_aug(t1, global_aug1)
            t1_v2 = apply_batch_aug(t1, global_aug2)
            t2_v1 = apply_batch_aug(t2, global_aug1)
            t2_v2 = apply_batch_aug(t2, global_aug2)

            t1_v1 = t1_v1.to(device, non_blocking=True)
            t1_v2 = t1_v2.to(device, non_blocking=True)
            t2_v1 = t2_v1.to(device, non_blocking=True)
            t2_v2 = t2_v2.to(device, non_blocking=True)

            B = t1_v1.size(0)
            num_chunks = math.ceil(B / student_chunk_size)

            # ===== teacher 整批前向 =====
            with torch.no_grad():
                teacher_out_t1 = model.teacher_forward(
                    t1_v1,
                    T=t_teacher,
                    return_logits=True,
                    return_features=True,
                    return_layer_tokens=True,
                    update_codebook=True,
                )
                teacher_out_t2 = model.teacher_forward(
                    t2_v1,
                    T=t_teacher,
                    return_logits=True,
                    return_features=True,
                    return_layer_tokens=True,
                    update_codebook=True,
                )

                q_t1 = teacher_out_t1["probs"]
                q_t2 = teacher_out_t2["probs"]
                logit_t1 = teacher_out_t1["logits"]
                logit_t2 = teacher_out_t2["logits"]

                token_ids_t1 = teacher_out_t1["layer_token_ids"]
                token_ids_t2 = teacher_out_t2["layer_token_ids"]

                conf_t1 = model.batch_confidence_from_probs(q_t1)
                conf_t2 = model.batch_confidence_from_probs(q_t2)

                cm_w_t1_from_t2 = conf_t2.pow(conf_power).detach()
                cm_w_t2_from_t1 = conf_t1.pow(conf_power).detach()

                mean_conf = 0.5 * (conf_t1.mean() + conf_t2.mean())

            opt.zero_grad(set_to_none=True)

            batch_loss_value = 0.0
            batch_loss_cm_value = 0.0
            batch_loss_sm_value = 0.0
            batch_loss_tok_value = 0.0
            batch_weighted_cm_value = 0.0
            batch_weighted_sm_value = 0.0
            batch_weighted_tok_value = 0.0

            # ===== student 分块反传 =====
            for start in range(0, B, student_chunk_size):
                end = min(start + student_chunk_size, B)

                s_t1_v1 = t1_v1[start:end]
                s_t1_v2 = t1_v2[start:end]
                s_t2_v1 = t2_v1[start:end]
                s_t2_v2 = t2_v2[start:end]

                q1 = q_t1[start:end]
                q2 = q_t2[start:end]
                w_cm_from_q2 = cm_w_t1_from_t2[start:end]
                w_cm_from_q1 = cm_w_t2_from_t1[start:end]

                # student outputs
                s_out_t1_v1 = model.student_forward_logits(
                    s_t1_v1, T=t_student, return_feat=True, return_layer_token_logits=True
                )
                s_out_t1_v2 = model.student_forward_logits(
                    s_t1_v2, T=t_student, return_feat=True, return_layer_token_logits=True
                )
                s_out_t2_v1 = model.student_forward_logits(
                    s_t2_v1, T=t_student, return_feat=True, return_layer_token_logits=True
                )
                s_out_t2_v2 = model.student_forward_logits(
                    s_t2_v2, T=t_student, return_feat=True, return_layer_token_logits=True
                )

                p_t1_v1 = s_out_t1_v1["log_probs"]
                p_t1_v2 = s_out_t1_v2["log_probs"]
                p_t2_v1 = s_out_t2_v1["log_probs"]
                p_t2_v2 = s_out_t2_v2["log_probs"]

                # ===== 1) same-modal anchor =====
                loss_sm = (
                    dino_loss(p_t1_v2, q1) +
                    dino_loss(p_t2_v2, q2)
                ) * 0.5

                # ===== 2) cross-modal fine alignment =====
                loss_cm = (
                    weighted_dino_loss(p_t1_v1, q2, w_cm_from_q2) +
                    weighted_dino_loss(p_t1_v2, q2, w_cm_from_q2) +
                    weighted_dino_loss(p_t2_v1, q1, w_cm_from_q1) +
                    weighted_dino_loss(p_t2_v2, q1, w_cm_from_q1)
                ) * 0.25

                # ===== 3) layer-wise discrete token distillation =====
                # 用 teacher 的 layer token id 作为目标，
                # student 在每层预测 token 分类。
                # 为了兼顾同模态稳定与跨模态共享表示，这里采用：
                # - 同模态 token distillation: t1_v2 <- t1 teacher, t2_v2 <- t2 teacher
                # - 跨模态 token distillation: t1_v1 <- t2 teacher, t2_v1 <- t1 teacher
                tok_sm = (
                    layer_token_ce_loss(
                        s_out_t1_v2["layer_token_logits"],
                        [x[start:end] for x in token_ids_t1],
                        layer_weights=token_layer_weights,
                    ) +
                    layer_token_ce_loss(
                        s_out_t2_v2["layer_token_logits"],
                        [x[start:end] for x in token_ids_t2],
                        layer_weights=token_layer_weights,
                    )
                ) * 0.5

                tok_cm = (
                    layer_token_ce_loss(
                        s_out_t1_v1["layer_token_logits"],
                        [x[start:end] for x in token_ids_t2],
                        layer_weights=token_layer_weights,
                    ) +
                    layer_token_ce_loss(
                        s_out_t2_v1["layer_token_logits"],
                        [x[start:end] for x in token_ids_t1],
                        layer_weights=token_layer_weights,
                    )
                ) * 0.5

                # token loss 也做成 anchor + cross-modal 的结构
                loss_tok = 0.5 * tok_sm + 0.5 * tok_cm

                loss = (
                    w_sm_epoch * loss_sm +
                    w_cm_epoch * loss_cm +
                    w_token_epoch * loss_tok
                )

                (loss / num_chunks).backward()

                batch_loss_value += float(loss.item())
                batch_loss_cm_value += float(loss_cm.item())
                batch_loss_sm_value += float(loss_sm.item())
                batch_loss_tok_value += float(loss_tok.item())
                batch_weighted_cm_value += float((w_cm_epoch * loss_cm).item())
                batch_weighted_sm_value += float((w_sm_epoch * loss_sm).item())
                batch_weighted_tok_value += float((w_token_epoch * loss_tok).item())

            nn.utils.clip_grad_norm_(model.student_backbone.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(model.student_head.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(model.student_layer_projectors.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(model.layer_token_predictors.parameters(), max_norm=5.0)

            opt.step()

            # ===== 更新 teacher 与 center =====
            with torch.no_grad():
                model.update_teacher(m=teacher_m)
                model.update_center_logits(torch.cat([logit_t1, logit_t2], dim=0), momentum=center_m)

            batch_loss_value /= max(1, num_chunks)
            batch_loss_cm_value /= max(1, num_chunks)
            batch_loss_sm_value /= max(1, num_chunks)
            batch_loss_tok_value /= max(1, num_chunks)
            batch_weighted_cm_value /= max(1, num_chunks)
            batch_weighted_sm_value /= max(1, num_chunks)
            batch_weighted_tok_value /= max(1, num_chunks)

            running += batch_loss_value
            running_cm += batch_loss_cm_value
            running_sm += batch_loss_sm_value
            running_tok += batch_loss_tok_value
            running_weighted_cm += batch_weighted_cm_value
            running_weighted_sm += batch_weighted_sm_value
            running_weighted_tok += batch_weighted_tok_value
            running_conf += float(mean_conf.item())

            t.set_postfix({
                "loss": f"{batch_loss_value:.5f}",
                "cm": f"{batch_loss_cm_value:.5f}",
                "sm": f"{batch_loss_sm_value:.5f}",
                "tok": f"{batch_loss_tok_value:.5f}",
                "conf": f"{float(mean_conf.item()):.3f}",
                "w_cm": f"{w_cm_epoch:.3f}",
                "w_sm": f"{w_sm_epoch:.3f}",
                "w_tok": f"{w_token_epoch:.3f}",
            })
            t.update(1)

    denom = max(1, len(loader))
    return {
        "loss": running / denom,
        "loss_cm": running_cm / denom,
        "loss_sm": running_sm / denom,
        "loss_tok": running_tok / denom,
        "loss_cm_weighted": running_weighted_cm / denom,
        "loss_sm_weighted": running_weighted_sm / denom,
        "loss_tok_weighted": running_weighted_tok / denom,
        "mean_conf": running_conf / denom,
        "w_cm_epoch": w_cm_epoch,
        "w_sm_epoch": w_sm_epoch,
        "w_token_epoch": w_token_epoch,
    }


# ======================
# 覆盖可视化辅助
# ======================
def to_3ch_image(img: np.ndarray, fallback_gray: np.ndarray) -> np.ndarray:
    if img is None:
        base = np.stack([fallback_gray, fallback_gray, fallback_gray], axis=2)
        return base
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    return np.stack([fallback_gray, fallback_gray, fallback_gray], axis=2)

def overlay_fp_fn(base_bgr: np.ndarray, fp_mask255: np.uint8, fn_mask255: np.uint8,
                  alpha_fp: float = 0.45, alpha_fn: float = 0.45):
    fp_layer = np.zeros_like(base_bgr)
    fn_layer = np.zeros_like(base_bgr)
    fp_layer[fp_mask255 > 0] = (0, 0, 255)
    fn_layer[fn_mask255 > 0] = (255, 0, 0)
    fp_vis = cv2.addWeighted(base_bgr, 1.0, fp_layer, alpha_fp, 0)
    fn_vis = cv2.addWeighted(base_bgr, 1.0, fn_layer, alpha_fn, 0)
    combined = cv2.addWeighted(fp_vis, 1.0, fn_layer, alpha_fn, 0)
    return combined, fp_vis, fn_vis

def overlay_tp_fp_fn(base_bgr: np.ndarray,
                     tp_mask255: np.uint8, fp_mask255: np.uint8, fn_mask255: np.uint8,
                     alpha_tp: float = 0.45, alpha_fp: float = 0.45, alpha_fn: float = 0.45):
    tp_layer = np.zeros_like(base_bgr)
    fp_layer = np.zeros_like(base_bgr)
    fn_layer = np.zeros_like(base_bgr)
    tp_layer[tp_mask255 > 0] = (0, 255, 0)
    fp_layer[fp_mask255 > 0] = (0, 0, 255)
    fn_layer[fn_mask255 > 0] = (255, 0, 0)
    tp_vis = cv2.addWeighted(base_bgr, 1.0, tp_layer, alpha_tp, 0)
    fp_vis = cv2.addWeighted(base_bgr, 1.0, fp_layer, alpha_fp, 0)
    fn_vis = cv2.addWeighted(base_bgr, 1.0, fn_layer, alpha_fn, 0)
    combined = cv2.addWeighted(base_bgr, 1.0, tp_layer, alpha_tp, 0)
    combined = cv2.addWeighted(combined, 1.0, fp_layer, alpha_fp, 0)
    combined = cv2.addWeighted(combined, 1.0, fn_layer, alpha_fn, 0)
    return combined, tp_vis, fp_vis, fn_vis


# ======================
# 验证
# ======================
@torch.no_grad()
def validate_change_map(model, loader, o_h, o_w, gt, vis_dir, device, x1_full=None, x2_full=None, threshold_mode="fixed", fixed_threshold=66.3):
    model.eval()

    res = []
    with tqdm(total=len(loader), desc='Validate (CM)', ncols=150, colour='cyan') as t:
        for (t1, t2, _) in loader:
            t1 = t1.to(device).float()
            t2 = t2.to(device).float()

            f1 = model.student_backbone(t1)
            f2 = model.student_backbone(t2)

            B = t1.size(0)
            d = l2_distance(f1.view(B, -1), f2.view(B, -1))
            res.extend(d.cpu().numpy().tolist())
            t.update(1)

    cmi = minmax01(np.array(res)).reshape(o_h, o_w)
    cmi_u8 = to_uint8_01(cmi)
    cv2.imwrite(os.path.join(vis_dir, 'CMI_DINO.png'), cmi_u8)

    if threshold_mode == "otsu":
        thr, bcm = cv2.threshold(
            cmi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif threshold_mode == "fixed":
        thr = float(fixed_threshold)
        bcm = (cmi_u8 > thr).astype(np.uint8) * 255
    else:
        raise ValueError("threshold_mode must be 'fixed' or 'otsu'.")
    cv2.imwrite(os.path.join(vis_dir, 'BCM_DINO.png'), bcm)

    evaler = Evaluation(gt.astype(np.uint8), bcm)
    OA, KC, AA = evaler.Classification_indicators()
    P, R, F1 = evaler.ObjectExtract_indicators()

    FPR, TPR, _ = metrics.roc_curve((gt > 127).astype(int).flatten(), cmi.flatten())
    AUC = metrics.auc(FPR, TPR)

    pred = (bcm > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    FP = ((pred == 1) & (gt_bin == 0)).astype(np.uint8) * 255
    FN = ((pred == 0) & (gt_bin == 1)).astype(np.uint8) * 255
    TP = ((pred == 1) & (gt_bin == 1)).astype(np.uint8) * 255

    base_raw = x2_full if x2_full is not None else x1_full
    base_bgr = to_3ch_image(base_raw, cmi_u8)

    H, W = base_bgr.shape[:2]
    if (FP.shape[0] != H) or (FP.shape[1] != W):
        FP = cv2.resize(FP, (W, H), interpolation=cv2.INTER_NEAREST)
        FN = cv2.resize(FN, (W, H), interpolation=cv2.INTER_NEAREST)
        TP = cv2.resize(TP, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay_all_2, overlay_fp, overlay_fn = overlay_fp_fn(base_bgr, FP, FN, alpha_fp=0.45, alpha_fn=0.45)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_FN.png"), overlay_all_2)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_only.png"), overlay_fp)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FN_only.png"), overlay_fn)

    overlay_all_3, overlay_tp, overlay_fp2, overlay_fn2 = overlay_tp_fp_fn(
        base_bgr, TP, FP, FN, alpha_tp=0.45, alpha_fp=0.45, alpha_fn=0.45
    )
    cv2.imwrite(os.path.join(vis_dir, "Overlay_TP_FP_FN.png"), overlay_all_3)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_TP_only.png"), overlay_tp)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_only_v2.png"), overlay_fp2)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FN_only_v2.png"), overlay_fn2)

    print(f'[VAL][CMSDL-MSDL] AUC={AUC:.4f} OA={OA:.2f} KC={KC:.2f} F1={F1:.2f}')
    return cmi_u8, bcm, dict(AUC=AUC, OA=OA, KC=KC, F1=F1, threshold=float(thr))


# ======================
# 实验日志、CSV 表格与曲线
# ======================
def write_csv(path, rows, fieldnames=None):
    """Write a list of dictionaries to CSV (UTF-8 with BOM for Excel)."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_run_plots(history, output_dir):
    """Save per-run training-loss, evaluation-metric, and constant-weight curves."""
    if not history:
        return
    if plt is None:
        print("[WARN] matplotlib is not installed; CSV/JSON logs were saved, but plots were skipped.")
        return

    epochs = [int(x["epoch"]) for x in history]

    # Training losses: objective, raw branch losses, and weighted contributions.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(epochs, [x["train_loss"] for x in history], marker="o", markersize=3)
    axes[0, 0].set_title("Total Training Objective")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    for key, label in [
        ("train_loss_cm", "Cross-modal (raw)"),
        ("train_loss_sm", "Same-modal (raw)"),
        ("train_loss_tok", "Token (raw)"),
    ]:
        axes[0, 1].plot(epochs, [x[key] for x in history], marker="o", markersize=2, label=label)
    axes[0, 1].set_title("Unweighted Loss Components")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for key, label in [
        ("train_loss_cm_weighted", "w_cm × L_cm"),
        ("train_loss_sm_weighted", "w_sm × L_sm"),
        ("train_loss_tok_weighted", "w_token × L_token"),
    ]:
        axes[1, 0].plot(epochs, [x[key] for x in history], marker="o", markersize=2, label=label)
    axes[1, 0].set_title("Weighted Loss Contributions")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Weighted loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    for key, label in [
        ("w_cm_epoch", "w_cm"),
        ("w_sm_epoch", "w_sm"),
        ("w_token_epoch", "w_token"),
    ]:
        axes[1, 1].plot(epochs, [x[key] for x in history], marker="o", markersize=2, label=label)
    axes[1, 1].set_title("Effective Loss Weights")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Weight")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_loss_curves.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Evaluation metrics. Separate axes avoid mixing AUC [0, 1] with percentage metrics.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("AUC", "AUC", "AUC"),
        ("OA", "Overall Accuracy (OA)", "OA"),
        ("KC", "Kappa Coefficient (KC)", "KC"),
        ("F1", "F1 Score", "F1"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, metric_specs):
        ax.plot(epochs, [x[key] for x in history], marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "evaluation_metrics_curves.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_history_artifacts(history, output_dir, args):
    """Persist logs after every epoch, so interrupted runs still retain results."""
    json_path = os.path.join(output_dir, "history_all.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    csv_rows = []
    for item in history:
        row = dict(item)
        row["weighting_strategy"] = "fixed_final"
        row["seed"] = args.seed
        row["threshold_mode"] = args.threshold_mode
        row["fixed_threshold"] = args.fixed_threshold
        csv_rows.append(row)
    write_csv(os.path.join(output_dir, "epoch_metrics.csv"), csv_rows)
    save_run_plots(history, output_dir)


def save_run_summary(history, output_dir, args):
    """Write final-epoch metrics and best-F1 epoch (diagnostic only)."""
    if not history:
        return
    final = history[-1]
    best = max(history, key=lambda x: float(x["F1"]))
    row = {
        "weighting_strategy": "fixed_final",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "threshold_mode": args.threshold_mode,
        "fixed_threshold": float(args.fixed_threshold),
        "w_cm_target": float(args.w_cm),
        "w_sm": float(args.w_sm),
        "w_token_target": float(args.w_token),
        "final_epoch": int(final["epoch"]),
        "final_AUC": float(final["AUC"]),
        "final_OA": float(final["OA"]),
        "final_KC": float(final["KC"]),
        "final_F1": float(final["F1"]),
        "best_f1_epoch_diagnostic_only": int(best["epoch"]),
        "best_f1_AUC_diagnostic_only": float(best["AUC"]),
        "best_f1_OA_diagnostic_only": float(best["OA"]),
        "best_f1_KC_diagnostic_only": float(best["KC"]),
        "best_f1_F1_diagnostic_only": float(best["F1"]),
    }
    write_csv(os.path.join(output_dir, "run_summary.csv"), [row])



# ======================
# 主程序
# ======================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', default='yellow', type=str)
    parser.add_argument('--t1_path', default='./data/Yellow/yellow_C_1.bmp', type=str)
    parser.add_argument('--t2_path', default='./data/Yellow/yellow_C_2.bmp', type=str)
    parser.add_argument('--gt_path', default='./data/Yellow/yellow_C_gt.bmp', type=str)

    parser.add_argument('--t1_nc', default=3, type=int)
    parser.add_argument('--t2_nc', default=1, type=int)
    parser.add_argument('--patch_size', default=11, type=int)
    parser.add_argument('--test_ps', default=11, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--vision_path', default='./runs/cmsdl_fixed_final_yellow/', type=str)
    parser.add_argument(
        '--threshold_mode', default='fixed', choices=['fixed', 'otsu'], type=str,
        help='fixed preserves the original code threshold; otsu follows Eq. (36) in the manuscript.'
    )
    parser.add_argument('--fixed_threshold', default=66.3, type=float)

    # DINO / CMSDL
    parser.add_argument('--out_dim', default=256, type=int)
    parser.add_argument('--head_hidden', default=512, type=int)
    parser.add_argument('--head_bottleneck', default=256, type=int)
    parser.add_argument('--t_teacher', default=0.04, type=float)
    parser.add_argument('--t_student', default=0.10, type=float)
    parser.add_argument('--teacher_m', default=0.996, type=float)
    parser.add_argument('--center_m', default=0.9, type=float)

    # CMSDL-MSD loss weights
    parser.add_argument('--w_cm', default=0.30, type=float)
    parser.add_argument('--w_sm', default=0.10, type=float)

    # DistilMOS-style layer token distillation
    parser.add_argument('--w_token', default=0.10, type=float)
    parser.add_argument('--token_dim', default=128, type=int)
    parser.add_argument('--token_hidden', default=256, type=int)
    parser.add_argument('--num_layer_tokens', default=64, type=int)
    parser.add_argument('--token_codebook_m', default=0.95, type=float)

    parser.add_argument('--student_chunk_size', default=128, type=int)
    parser.add_argument('--conf_power', default=1.0, type=float)
    parser.add_argument('--seed', default=2025, type=int)

    args = parser.parse_args()

    if args.batch_size <= 0 or args.student_chunk_size <= 0:
        parser.error("batch_size and student_chunk_size must be positive.")
    if args.epochs <= 0:
        parser.error("epochs must be positive.")

    set_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.vision_path)

    config = vars(args).copy()
    config["weighting_strategy"] = "fixed_final"
    config["effective_weights"] = {
        "w_cm": float(args.w_cm),
        "w_sm": float(args.w_sm),
        "w_token": float(args.w_token),
    }
    with open(os.path.join(args.vision_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    train_dataset = Data_Loader(
        args.data_name, args.t1_path, args.t2_path, args.gt_path,
        patch_size=args.patch_size, mode='train', transform=T.ToTensor()
    )
    test_dataset = Data_Loader(
        args.data_name, args.t1_path, args.t2_path, args.gt_path,
        patch_size=args.test_ps, mode='test', transform=T.ToTensor()
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 6,
        shuffle=False,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    if args.data_name == 'bastrop':
        mat = io.loadmat(args.t1_path)
        x1 = mat['t1_L5'][:, :, 3]
        x2 = mat["t2_ALI"][:, :, 5]
        x1 = to_normalization(x1)[..., np.newaxis]
        x2 = to_normalization(x2)[..., np.newaxis]
        gt = (mat["ROI_1"] * 255).astype(np.uint8)
        x1_full = (x1 * 255).astype(np.uint8)
        x2_full = (x2 * 255).astype(np.uint8)
        x1_full = np.repeat(x1_full, 3, axis=2)
        x2_full = np.repeat(x2_full, 3, axis=2)
    else:
        x1_full = cv2.imread(args.t1_path)
        x2_full = cv2.imread(args.t2_path)
        gt = cv2.imread(args.gt_path)[:, :, 0].astype(np.uint8)

    o_h, o_w = gt.shape[:2]

    model = DINOStudentTeacher(
        in_channels=3,
        feat_dim=64,
        head_hidden=args.head_hidden,
        head_bottleneck=args.head_bottleneck,
        out_dim=args.out_dim,
        teacher_momentum=args.teacher_m,
        token_dim=args.token_dim,
        token_hidden=args.token_hidden,
        num_layer_tokens=args.num_layer_tokens,
        token_codebook_m=args.token_codebook_m,
    ).to(device)

    opt = optim.RMSprop(
        [
            {"params": model.student_backbone.parameters(), "lr": args.lr},
            {"params": model.student_head.parameters(), "lr": args.lr},
            {"params": model.student_layer_projectors.parameters(), "lr": args.lr},
            {"params": model.layer_token_predictors.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=1e-5,
        momentum=0.9,
    )

    # 层权重：可让深层略强一些
    token_layer_weights = [0.7, 0.9, 1.1, 1.3]

    history = []

    print("\n========== Fixed-final Weighting ==========")
    print(f"w_cm={args.w_cm:.4f}, w_sm={args.w_sm:.4f}, w_token={args.w_token:.4f}")
    print(f"These weights remain constant for all {args.epochs} epochs.")
    print(f"Threshold mode: {args.threshold_mode}; fixed threshold: {args.fixed_threshold}")
    print("===========================================\n")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            opt=opt,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            t_teacher=args.t_teacher,
            t_student=args.t_student,
            dino_w_cm=args.w_cm,
            dino_w_sm=args.w_sm,
            dino_w_token=args.w_token,
            center_m=args.center_m,
            teacher_m=args.teacher_m,
            student_chunk_size=args.student_chunk_size,
            conf_power=args.conf_power,
            token_layer_weights=token_layer_weights,
        )

        # Sanity checks: do not silently accept invalid or collapsed loss values.
        for loss_key in ("loss", "loss_cm", "loss_sm", "loss_tok"):
            if not np.isfinite(train_stats[loss_key]):
                raise FloatingPointError(
                    f"Non-finite {loss_key} at epoch {epoch}: {train_stats[loss_key]}"
                )
        if train_stats["loss_tok"] <= 1e-12:
            print(
                f"[WARNING] Token distillation loss is approximately zero at epoch {epoch}. "
                "Please inspect teacher token IDs, codebook assignments, and "
                "layer_token_ce_loss()."
            )

        print(
            f"[EPOCH {epoch}] "
            f"loss={train_stats['loss']:.5f} "
            f"cm={train_stats['loss_cm']:.5f} "
            f"sm={train_stats['loss_sm']:.5f} "
            f"tok={train_stats['loss_tok']:.5f} "
            f"conf={train_stats['mean_conf']:.4f} "
            f"w_cm={train_stats['w_cm_epoch']:.4f} "
            f"w_sm={train_stats['w_sm_epoch']:.4f} "
            f"w_tok={train_stats['w_token_epoch']:.4f}"
        )

        ep_dir = os.path.join(args.vision_path, str(epoch))
        ensure_dir(ep_dir)

        cmi_u8, bcm_u8, metrics_dict = validate_change_map(
            model=model,
            loader=test_loader,
            o_h=o_h,
            o_w=o_w,
            gt=gt,
            vis_dir=ep_dir,
            device=device,
            x1_full=x1_full,
            x2_full=x2_full,
            threshold_mode=args.threshold_mode,
            fixed_threshold=args.fixed_threshold,
        )

        io.savemat(
            os.path.join(ep_dir, 'DINO_outputs.mat'),
            {'cmi_u8': cmi_u8, 'bcm_u8': bcm_u8}
        )

        record = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_loss_cm": float(train_stats["loss_cm"]),
            "train_loss_sm": float(train_stats["loss_sm"]),
            "train_loss_tok": float(train_stats["loss_tok"]),
            "train_loss_cm_weighted": float(train_stats["loss_cm_weighted"]),
            "train_loss_sm_weighted": float(train_stats["loss_sm_weighted"]),
            "train_loss_tok_weighted": float(train_stats["loss_tok_weighted"]),
            "weighting_strategy": "fixed_final",
            "seed": int(args.seed),
            "mean_conf": float(train_stats["mean_conf"]),
            "w_cm_epoch": float(train_stats["w_cm_epoch"]),
            "w_sm_epoch": float(train_stats["w_sm_epoch"]),
            "w_token_epoch": float(train_stats["w_token_epoch"]),
            **{k: float(v) for k, v in metrics_dict.items()},
        }
        history.append(record)

        with open(os.path.join(ep_dir, 'metrics_CMSDL_MSDL.json'), 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # Update CSV/JSON tables and curves after every epoch for fault tolerance.
        save_history_artifacts(history, args.vision_path, args)

        torch.save(
            model.state_dict(),
            os.path.join(ep_dir, f'cmsdl_msdl_epoch{epoch}.pth')
        )

    save_history_artifacts(history, args.vision_path, args)
    save_run_summary(history, args.vision_path, args)
    torch.save(model.state_dict(), os.path.join(args.vision_path, "final_model.pth"))

    print(f"Training finished. Results saved to: {args.vision_path}")
    print("Key files: epoch_metrics.csv, run_summary.csv, training_loss_curves.png, evaluation_metrics_curves.png, final_model.pth")

if __name__ == '__main__':
    main()
