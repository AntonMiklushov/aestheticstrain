#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт собирает *минимальные и максимальные значения raw_mean_score*,
которые модель выдаёт на TRAIN части AVA.

Это "настоящий" диапазон, на котором модель обучалась, без перцентилей.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader


# ========= Модель (как в finetune.py) =========

class ResNet50AvaMultiHead(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = backbone.fc.in_features

        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.fc_dist = nn.Linear(in_features, 10)
        self.fc_ord = nn.Linear(in_features, 9)

    def forward(self, x):
        features = self.backbone(x)
        logits_dist = self.fc_dist(features)
        logits_ord = self.fc_ord(features)
        return logits_dist, logits_ord


# ========= AVA utility =========

def read_ava_txt(path: Path) -> pd.DataFrame:
    cols = ["image_id"] + [f"n{k}" for k in range(1, 11)]
    df = pd.read_csv(
        path, sep=" ", header=None, usecols=range(1, 12), names=cols
    )
    vote_cols = [f"n{k}" for k in range(1, 11)]
    df["n_total"] = df[vote_cols].sum(axis=1).clip(lower=1)

    weights = np.arange(1, 11, dtype=np.float32)
    df["mean_score"] = (df[vote_cols].values * weights).sum(axis=1) / df["n_total"]

    df["image_id"] = df["image_id"].astype(str)
    return df


def read_id_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["image_id"])
    df["image_id"] = df["image_id"].astype(str)
    return df


# ========= Dataset =========

class AVATrainDataset(Dataset):
    """Простой датасет: отдаёт только изображение."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.images_dir / f"{row['image_id']}.jpg"

        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (448, 448), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img


# ========= Трансформация =========

def get_eval_transform(size=448):
    return T.Compose([
        T.Resize(int(size * 1.143)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


# ========= Основная логика =========

@torch.no_grad()
def collect_train_minmax(model, loader, device):
    raw_scores = []

    model.eval()
    ks = torch.arange(1, 11, device=device).view(1, -1).float()

    for batch in loader:
        batch = batch.to(device)
        logits_dist, _ = model(batch)
        probs = F.softmax(logits_dist, dim=1)

        means = (probs * ks).sum(dim=1)
        raw_scores.append(means.cpu().numpy())

    raw_scores = np.concatenate(raw_scores)

    return raw_scores.min(), raw_scores.max()


def main():
    # Пути под ваш проект
    project = Path(".")
    model_path = project / "models" / "resnet50_ava_regression_finetune_001.pt"
    ava_root = project / "data" / "AVA"

    ava_txt = ava_root / "AVA.txt"
    train_list = ava_root / "aesthetics_image_lists" / "generic_ls_train.jpgl"
    images_dir = ava_root / "images"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading AVA…")
    ava = read_ava_txt(ava_txt)
    train_ids = read_id_list(train_list)
    train_df = ava[ava["image_id"].isin(train_ids["image_id"])]

    print(f"Train size: {len(train_df)}")

    ds = AVATrainDataset(train_df, images_dir, transform=get_eval_transform(448))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    print("Loading model…")
    model = ResNet50AvaMultiHead(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    print("Computing RAW_MIN / RAW_MAX on TRAIN…")
    raw_min, raw_max = collect_train_minmax(model, loader, device)

    print("\n======= RESULT =======")
    print(f"RAW_MIN = {raw_min:.4f}")
    print(f"RAW_MAX = {raw_max:.4f}")
    print("======================\n")

    print("Подставьте эти значения в нормализацию в Gradio/Android/Kivy.")


if __name__ == "__main__":
    main()
