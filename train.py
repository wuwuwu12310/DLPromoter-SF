import os
import re
import time
import json
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, max_error, explained_variance_score
)
from scipy.stats import pearsonr, spearmanr

from model import TransformerHybridSEFusionModel


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_seq(s: str) -> str:
    s = s.strip().lower()
    s = re.sub("[^acgt]", "z", s)
    return s


def one_hot_encode(seq: str):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    arr = np.array(list(seq), dtype="<U1")
    idx = np.array([mapping.get(c, 4) for c in arr], dtype=np.int64)
    oh = np.zeros((len(idx), 4), dtype=np.float32)
    mask = idx < 4
    oh[np.arange(len(idx))[mask], idx[mask]] = 1.0
    return oh


def pad_or_trunc(mat: np.ndarray, max_len: int):
    L = mat.shape[0]
    if L > max_len:
        return mat[:max_len]
    if L < max_len:
        pad_len = max_len - L
        return np.pad(mat, ((0, pad_len), (0, 0)))
    return mat


def load_csv(path: str):
    seqs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        _ = next(f, None)
        for line in f:
            if not line.strip():
                continue
            s, y = line.strip().split(",")
            seqs.append(s)
            ys.append(float(y))
    return seqs, np.asarray(ys, dtype=np.float32)


def torch_load_state_dict(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -----------------------
# Extra features (76)
# -----------------------
_BASES = ["a", "c", "g", "t"]
_KMER3 = [a + b + c for a in _BASES for b in _BASES for c in _BASES]
_KMER3_TO_IDX = {k: i for i, k in enumerate(_KMER3)}


def kmer3_freq(seq: str):
    seq = seq.lower()
    counts = np.zeros((64,), dtype=np.float32)
    denom = 0
    for i in range(len(seq) - 2):
        k = seq[i:i + 3]
        if "z" in k:
            continue
        counts[_KMER3_TO_IDX[k]] += 1.0
        denom += 1
    if denom > 0:
        counts /= denom
    return counts


def gc_profile(seq: str, n_bins: int = 8):
    seq = seq.lower()
    L = len(seq)
    out = np.zeros((n_bins,), dtype=np.float32)
    for b in range(n_bins):
        l = int(round(b * L / n_bins))
        r = int(round((b + 1) * L / n_bins))
        chunk = seq[l:r]
        valid = [c for c in chunk if c in "acgt"]
        if len(valid) == 0:
            out[b] = 0.0
            continue
        gc = sum(1 for c in valid if c in "gc") / float(len(valid))
        out[b] = gc
    return out


def global_comp(seq: str):
    seq = seq.lower()
    valid = [c for c in seq if c in "acgt"]
    if len(valid) == 0:
        return np.zeros((2,), dtype=np.float32)
    gc = sum(1 for c in valid if c in "gc") / float(len(valid))
    at = 1.0 - gc
    return np.asarray([gc, at], dtype=np.float32)


def longest_homopolymer(seq: str, base: str):
    seq = seq.lower()
    best = cur = 0
    for c in seq:
        if c == base:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best)


def build_extra_features(seq: str, n_bins: int = 8):
    seq = clean_seq(seq)
    f1 = kmer3_freq(seq)  # 64
    f2 = gc_profile(seq, n_bins=n_bins)  # 8
    f3 = global_comp(seq)  # 2
    f4 = np.asarray([
        longest_homopolymer(seq, "a"),
        longest_homopolymer(seq, "t")
    ], dtype=np.float32)  # 2
    return np.concatenate([f1, f2, f3, f4], axis=0).astype(np.float32)  # 76


# -----------------------
# Metrics
# -----------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.float64).ravel()
    y_pred = y_pred.astype(np.float64).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mxerr = float(max_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    ev = float(explained_variance_score(y_true, y_pred))
    pr = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0
    sr = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "MaxError": mxerr,
        "MSE": float(mse),
        "ExplainedVar": ev,
        "Pearson": pr,
        "Spearman": sr,
    }


# -----------------------
# Losses
# -----------------------
class WeightedMSELoss(nn.Module):
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.register_buffer("w", torch.tensor([weight], dtype=torch.float32), persistent=False)

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        return torch.mean(self.w * (y_true - y_pred) ** 2)


class TopKHuberTail(nn.Module):
    """
    Stage2：MSE + alpha * mean(topk SmoothL1)
    """

    def __init__(self, alpha: float = 0.12, topk_frac: float = 0.08, beta: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.topk_frac = float(topk_frac)
        self.huber = nn.SmoothL1Loss(beta=float(beta), reduction="none")

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        mse = torch.mean((pred - target) ** 2)

        hub = self.huber(pred, target)
        k = max(1, int(hub.numel() * self.topk_frac))
        top = torch.topk(hub, k, largest=True).values.mean()

        return mse + self.alpha * top


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {}
        self.backup = {}
        self.copy_from(model)

    @torch.no_grad()
    def copy_from(self, model: nn.Module):
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        msd = model.state_dict()

        for k, v in msd.items():
            if not torch.is_floating_point(v):
                self.shadow[k].copy_(v)
            else:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for k, v in model.state_dict().items():
            self.backup[k] = v.detach().clone()
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        model.load_state_dict(self.backup, strict=True)
        self.backup = {}


@contextmanager
def ema_scope(model: nn.Module, ema: ModelEMA = None):
    if ema is None:
        yield
    else:
        ema.apply_to(model)
        try:
            yield
        finally:
            ema.restore(model)


# -----------------------
# Trainer
# -----------------------
class Trainer95TryEMA:
    def __init__(
            self,
            train_csv,
            save_root,
            run_name="newdata",
            seed=42,
            gpu_id=1,
            val_ratio=0.1,
            batch_size=256,
            hidden_size=256,
            dropout_rate=0.2,
            lr_s1=1e-3,
            wd=1e-3,
            epochs_s1=160,
            patience_s1=25,
            lr_s2=1e-4,
            epochs_s2=80,
            patience_s2=15,
            lr_s3=3e-5,
            epochs_s3=40,
            patience_s3=10,
            use_amp=True,
            num_workers=2,
            gc_bins=8,
            normalize_extra=True,
            conv_kernels=(17, 13, 9),
            use_ema=True,
            ema_decay=0.999,
    ):
        set_seed(seed)

        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        ts = time.strftime("%Y%m%d_%H%M%S")
        ktag = f"k{conv_kernels[0]}_{conv_kernels[1]}_{conv_kernels[2]}"
        self.run_dir = os.path.join(save_root, f"{run_name}_{ktag}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.best_path = os.path.join(self.run_dir, "best_model.pth")
        self.log_path = os.path.join(self.run_dir, "train_log.txt")
        self.cfg_path = os.path.join(self.run_dir, "run_config.json")
        self.final_metrics_path = os.path.join(self.run_dir, "val_metrics.json")

        # ---- load data
        seq_train, y_train = load_csv(train_csv)
        train_oh_list = [one_hot_encode(clean_seq(s)) for s in seq_train]

        self.max_len = max(m.shape[0] for m in train_oh_list)

        X_train = np.stack([pad_or_trunc(m, self.max_len) for m in train_oh_list], axis=0)
        F_train = np.stack([build_extra_features(s, n_bins=gc_bins) for s in seq_train], axis=0)

        idx = np.arange(len(y_train))
        tr_idx, va_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, shuffle=True)

        self.extra_norm = None
        if normalize_extra:
            mu = F_train[tr_idx].mean(axis=0, keepdims=True)
            sd = F_train[tr_idx].std(axis=0, keepdims=True) + 1e-6
            F_train = (F_train - mu) / sd

            # 这里保存了训练集的均值和方差，供测试集使用
            self.extra_norm = {"mu": mu.flatten().tolist(), "sd": sd.flatten().tolist()}
            with open(os.path.join(self.run_dir, "extra_norm.json"), "w", encoding="utf-8") as f:
                json.dump(self.extra_norm, f, ensure_ascii=False, indent=2)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        F_train_t = torch.tensor(F_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        pin = bool(torch.cuda.is_available())
        self.train_loader = DataLoader(
            TensorDataset(X_train_t[tr_idx], F_train_t[tr_idx], y_train_t[tr_idx]),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
        )
        self.val_loader = DataLoader(
            TensorDataset(X_train_t[va_idx], F_train_t[va_idx], y_train_t[va_idx]),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
        )

        # ---- model
        self.model = TransformerHybridSEFusionModel(
            input_size=4,
            hidden_size=hidden_size,
            output_size=1,
            dropout_rate=dropout_rate,
            extra_feat_dim=76,
            use_extra=True,
            conv_kernels=conv_kernels,
        ).to(self.device)

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema = ModelEMA(self.model, decay=self.ema_decay) if self.use_ema else None

        self.epochs_s1, self.patience_s1 = int(epochs_s1), int(patience_s1)
        self.epochs_s2, self.patience_s2 = int(epochs_s2), int(patience_s2)
        self.epochs_s3, self.patience_s3 = int(epochs_s3), int(patience_s3)

        self.opt_s1 = torch.optim.Adam(self.model.parameters(), lr=float(lr_s1), weight_decay=float(wd))
        self.opt_s2 = torch.optim.Adam(self.model.head_parameters(), lr=float(lr_s2), weight_decay=float(wd))
        self.opt_s3 = torch.optim.Adam(self.model.parameters(), lr=float(lr_s3), weight_decay=float(wd))

        self.loss_s1 = WeightedMSELoss(weight=10.0).to(self.device)
        self.loss_s2 = TopKHuberTail(alpha=0.12, topk_frac=0.08, beta=1.0).to(self.device)
        self.loss_s3 = nn.MSELoss().to(self.device)

        self.use_amp = bool(use_amp and torch.cuda.is_available())
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.run_cfg = {
            "run_dir": self.run_dir,
            "device": str(self.device),
            "train_csv": train_csv,
            "max_len": int(self.max_len),
            "hidden_size": int(hidden_size),
            "dropout_rate": float(dropout_rate),
            "extra_feat_dim": 76,
            "use_extra": True,
            "gc_bins": int(gc_bins),
            "normalize_extra": bool(normalize_extra),
            "conv_kernels": list(conv_kernels),
            "use_ema": bool(use_ema),
            "ema_decay": float(ema_decay),
            "S1": {"lr": lr_s1, "epochs": self.epochs_s1, "patience": self.patience_s1},
            "S2": {"lr": lr_s2, "epochs": self.epochs_s2, "patience": self.patience_s2},
            "S3": {"lr": lr_s3, "epochs": self.epochs_s3, "patience": self.patience_s3},
        }
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.run_cfg, f, ensure_ascii=False, indent=2)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.run_cfg, ensure_ascii=False, indent=2) + "\n\n")

    @torch.no_grad()
    def _evaluate_current_model(self, loader):
        self.model.eval()
        preds, trues = [], []
        for x, feat, y in loader:
            x = x.to(self.device, non_blocking=True)
            feat = feat.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).view(-1)

            out = self.model(x, feat).view(-1)
            preds.append(out.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

        y_true = np.concatenate(trues).ravel()
        y_pred = np.concatenate(preds).ravel()
        return compute_metrics(y_true, y_pred), y_true, y_pred

    @torch.no_grad()
    def evaluate(self, loader, use_ema=False):
        if use_ema and self.ema is not None:
            with ema_scope(self.model, self.ema):
                metrics, y_true, y_pred = self._evaluate_current_model(loader)
        else:
            metrics, y_true, y_pred = self._evaluate_current_model(loader)
        return metrics, y_true, y_pred

    def _save_best_state(self, path):
        if self.use_ema and self.ema is not None:
            with ema_scope(self.model, self.ema):
                torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def _sync_ema_to_model(self):
        if self.use_ema and self.ema is not None:
            self.ema.copy_from(self.model)

    def _train_stage(self, opt, loss_fn, epochs, patience, name, freeze_backbone: bool):
        # freeze backbone or not
        for p in self.model.backbone_parameters():
            p.requires_grad = (not freeze_backbone)
        for p in self.model.head_parameters():
            p.requires_grad = True

        best_r2 = -1e18
        bad = 0
        best_path = os.path.join(self.run_dir, f"best_{name}.pth")

        for ep in range(epochs):
            self.model.train()
            total = 0.0

            for x, feat, y in self.train_loader:
                x = x.to(self.device, non_blocking=True)
                feat = feat.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True).view(-1)

                opt.zero_grad(set_to_none=True)

                if hasattr(torch, "amp"):
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        out = self.model(x, feat).view(-1)
                        loss = loss_fn(out, y)
                else:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        out = self.model(x, feat).view(-1)
                        loss = loss_fn(out, y)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(opt)
                self.scaler.update()

                if self.use_ema and self.ema is not None:
                    self.ema.update(self.model)

                total += float(loss.item())

            train_loss = total / max(1, len(self.train_loader))

            val_m, _, _ = self.evaluate(self.val_loader, use_ema=self.use_ema)

            improved = val_m["R2"] > best_r2
            if improved:
                best_r2 = val_m["R2"]
                bad = 0
                self._save_best_state(best_path)
            else:
                bad += 1

            line = (
                f"[{name} ep={ep:03d}] loss={train_loss:.6f} | "
                f"VAL R2={val_m['R2']:.6f} RMSE={val_m['RMSE']:.6f} MAE={val_m['MAE']:.6f} "
                f"MedAE={val_m['MedAE']:.6f} MaxE={val_m['MaxError']:.6f} | "
                f"MSE={val_m['MSE']:.6f} EV={val_m['ExplainedVar']:.6f} "
                f"P={val_m['Pearson']:.6f} S={val_m['Spearman']:.6f} | "
                f"best={best_r2:.6f} {'*' if improved else ''} "
                f"| EMA={'ON' if self.use_ema else 'OFF'}"
            )
            print(line)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            if bad >= patience:
                print(f"Early stop {name} at ep={ep}, best_R2={best_r2:.6f}")
                break

        self.model.load_state_dict(torch_load_state_dict(best_path, map_location=self.device))
        self._sync_ema_to_model()
        return best_path, best_r2

    def train(self):
        # S1
        p1, _ = self._train_stage(
            self.opt_s1, self.loss_s1,
            self.epochs_s1, self.patience_s1,
            "S1_full", freeze_backbone=False
        )
        val1, _, _ = self.evaluate(self.val_loader, use_ema=False)

        # S2
        p2, _ = self._train_stage(
            self.opt_s2, self.loss_s2,
            self.epochs_s2, self.patience_s2,
            "S2_tail_head", freeze_backbone=True
        )
        val2, _, _ = self.evaluate(self.val_loader, use_ema=False)

        # S3
        p3, _ = self._train_stage(
            self.opt_s3, self.loss_s3,
            self.epochs_s3, self.patience_s3,
            "S3_unfreeze_tiny", freeze_backbone=False
        )
        val3, _, _ = self.evaluate(self.val_loader, use_ema=False)

        # choose best by VAL R2
        cands = [("S1", p1, val1), ("S2", p2, val2), ("S3", p3, val3)]
        chosen = max(cands, key=lambda x: x[2]["R2"])
        name, path, val_best = chosen

        self.model.load_state_dict(torch_load_state_dict(path, map_location=self.device))
        self._sync_ema_to_model()
        torch.save(self.model.state_dict(), self.best_path)

        final_info = {
            "CONV_KERNELS": list(self.model.conv_kernels),
            "CHOSEN_STAGE": name,
            "VAL": val_best,
            "best_model_path": self.best_path
        }
        with open(self.final_metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_info, f, ensure_ascii=False, indent=2)

        print("\n===== BEST MODEL METRICS (VALIDATION) =====")
        print(f"CONV_KERNELS: {self.model.conv_kernels}")
        print(f"CHOSEN: {name}")
        print("VAL :", json.dumps(val_best, ensure_ascii=False))
        print(f"\nRun dir: {self.run_dir}")
        print(f"Best model: {self.best_path}")

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n===== BEST MODEL METRICS (VALIDATION) =====\n")
            f.write(f"CONV_KERNELS: {self.model.conv_kernels}\n")
            f.write(f"CHOSEN: {name}\n")
            f.write("VAL : " + json.dumps(val_best, ensure_ascii=False) + "\n")

        return val_best, self.run_dir


if __name__ == "__main__":
    TRAIN_CSV = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Data/SC/wrcprocess_train_data162982.csv"
    SAVE_ROOT = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/wrc_63468huigui/results_162982/"

    KERNEL_SETS = [
        # (17, 13, 9),
        (13, 11, 9),
        # (13, 9, 5),
    ]

    summary = []
    for ks in KERNEL_SETS:
        print("\n==============================")
        print("Running conv_kernels =", ks)

        trainer = Trainer95TryEMA(
            train_csv=TRAIN_CSV,
            save_root=SAVE_ROOT,
            run_name="high_similarity_train",
            seed=42,
            gpu_id=1,
            val_ratio=0.1,
            batch_size=256,
            hidden_size=256,
            dropout_rate=0.2,
            lr_s1=1e-3, wd=1e-3, epochs_s1=160, patience_s1=25,
            lr_s2=1e-4, epochs_s2=80, patience_s2=15,
            lr_s3=3e-5, epochs_s3=40, patience_s3=10,
            normalize_extra=True,
            use_amp=True,
            num_workers=2,
            gc_bins=8,
            conv_kernels=ks,
            use_ema=True,
            ema_decay=0.999,
        )
        val_m, run_dir = trainer.train()
        summary.append({
            "conv_kernels": list(ks),
            "VAL_R2": float(val_m["R2"]),
            "run_dir": run_dir,
        })

    summary = sorted(summary, key=lambda x: x["VAL_R2"], reverse=True)

    print("\n=========== SUMMARY (sorted by VAL R2) ===========")
    for item in summary:
        print(
            f"kernels={tuple(item['conv_kernels'])} | "
            f"VAL_R2={item['VAL_R2']:.6f} | "
            f"run_dir={item['run_dir']}"
        )

    summary_path = os.path.join(SAVE_ROOT, "summary_all_kernels_val.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary to: {summary_path}")
