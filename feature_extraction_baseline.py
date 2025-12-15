from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler

import finetune as ft

# Базовое имя для чекпоинта и метрик feature-extraction варианта.
FEATURE_BASENAME = "resnet50_ava_regression_feature"

# Количество эпох и LR можно держать скромными — модель обучается только головой.
NUM_EPOCHS = 10
LR = 2e-4


def make_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создаёт DataLoader'ы, повторяя логику finetune.py."""
    train_ds = ft.AVADistributionDataset(ft.train_df, transform=ft.get_train_transform())
    train_weights = torch.tensor(ft.train_df["sample_weight"].values, dtype=torch.double)
    train_sampler = WeightedRandomSampler(
        train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=ft.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=ft.NUM_WORKERS_TRAIN,
        pin_memory=True,
        prefetch_factor=ft.PREFETCH_FACTOR,
    )

    val_ds = ft.AVADistributionDataset(ft.val_df, transform=ft.get_eval_transform())
    val_loader = DataLoader(
        val_ds,
        batch_size=ft.BATCH_SIZE,
        shuffle=False,
        num_workers=ft.NUM_WORKERS_EVAL,
        pin_memory=True,
    )

    test_ds = ft.AVADistributionDataset(ft.test_df, transform=ft.get_eval_transform())
    test_loader = DataLoader(
        test_ds,
        batch_size=ft.BATCH_SIZE,
        shuffle=False,
        num_workers=ft.NUM_WORKERS_EVAL,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_feature_model(device: torch.device) -> torch.nn.Module:
    """ResNet-50 с замороженным backbone: тренируем только головы → худший baseline."""
    model = ft.ResNet50AvaMultiHead(pretrained=True).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model


def main():
    ft.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = make_loaders()
    model = build_feature_model(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=ft.WEIGHT_DECAY)
    scaler = GradScaler(device="cuda") if device.type == "cuda" else None

    best_state = None
    best_val_mse = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = ft.train_one_epoch(model, train_loader, device, optimizer, scaler)
        val_metrics = ft.evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mse": val_metrics.mse,
                "val_rmse": val_metrics.rmse,
                "val_mae": val_metrics.mae,
            }
        )
        if val_metrics.mse < best_val_mse:
            best_val_mse = val_metrics.mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_MSE={val_metrics.mse:.4f} | val_RMSE={val_metrics.rmse:.4f} | "
            f"val_MAE={val_metrics.mae:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = ft.evaluate_model(model, val_loader, device)
    test_metrics = ft.evaluate_model(model, test_loader, device)
    total_time = time.time() - t0

    models_dir = ft.MODELS_DIR
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f"{FEATURE_BASENAME}.pt"
    torch.save(model.state_dict(), model_path)

    metrics_path = models_dir / f"{FEATURE_BASENAME}_metrics.json"
    payload = {
        "model_path": str(model_path),
        "device": str(device),
        "train_params": {
            "epochs": NUM_EPOCHS,
            "batch_size": ft.BATCH_SIZE,
            "lr": LR,
            "weight_decay": ft.WEIGHT_DECAY,
            "tail_low_threshold": ft.TAIL_LOW_THRESHOLD,
            "tail_high_threshold": ft.TAIL_HIGH_THRESHOLD,
            "amp_enabled": device.type == "cuda",
            "duration_sec": total_time,
        },
        "history": history,
        "val": asdict(val_metrics),
        "test": asdict(test_metrics),
    }

    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved feature-extraction metrics to {metrics_path}")


if __name__ == "__main__":
    main()
