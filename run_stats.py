from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import numpy as np

# Импортируем всё нужное из вашего finetune.py
import finetune as ft


def get_latest_finetuned_model_path(models_dir: Path, basename: str) -> Path:
    """Находит последний по номеру чекпоинт вида
    basename_XXX.pt в указанной директории.

    Если файл не найден — кидает RuntimeError.
    """
    candidates = sorted(models_dir.glob(f"{basename}_*.pt"))
    if not candidates:
        raise RuntimeError(
            f"В {models_dir} не найдено ни одного файла {basename}_XXX.pt. "
            "Сначала обучите модель (finetune.py)."
        )
    # Берём последний в лексикографическом порядке, т.к. XXX форматирован с нулями.
    return candidates[-1]


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Загружает модель ResNet50AvaMultiHead с указанного чекпоинта."""
    model = ft.ResNet50AvaMultiHead(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def create_eval_loader(df, batch_size: int = 128) -> torch.utils.data.DataLoader:
    """Создаёт DataLoader для оценки (без аугментаций, без sampler'ов)."""
    ds = ft.AVADistributionDataset(df, transform=ft.get_eval_transform())
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Прогоняет датасет через модель и собирает:

    - all_preds: предсказанные mean_score (ожидание по распределению);
    - all_targets: истинные mean_score из AVA.

    Возвращает два тензора на CPU формы [N].
    """
    all_preds = []
    all_targets = []

    for images, target_dists, target_means, sample_weights in loader:
        images = images.to(device)
        target_means = target_means.to(device)

        # Модель возвращает (logits_dist, logits_ord),
        # но для статистики нам нужен только logits_dist.
        logits_dist, logits_ord = model(images)

        log_probs = torch.nn.functional.log_softmax(logits_dist, dim=1)
        probs = log_probs.exp()

        ks = torch.arange(1, 11, device=device, dtype=probs.dtype).view(1, -1)
        pred_means = (probs * ks).sum(dim=1)  # [B]

        all_preds.append(pred_means.cpu())
        all_targets.append(target_means.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets


def describe_array(x: np.ndarray, name: str) -> None:
    """Печатает простую сводную статистику для массива значений."""
    print(f"{name}:")
    print(f"  count = {len(x)}")
    print(f"  min   = {x.min():.4f}")
    print(f"  max   = {x.max():.4f}")
    print(f"  mean  = {x.mean():.4f}")
    print(f"  std   = {x.std():.4f}")
    print()


def describe_bins(
    preds: np.ndarray,
    targets: np.ndarray,
    name: str,
) -> None:
    """Печатает статистику по бинам:

    Бины по таргету:
      - [1, 3]
      - (3, 5]
      - (5, 7]
      - (7, 10]

    Для каждого бина выводится:
      - сколько примеров попало;
      - средний таргет;
      - среднее предсказание;
      - средняя абсолютная ошибка.
    """
    print(f"=== {name}: статистика по бинам (по таргету) ===")

    bins = [
        (1.0, 3.0, "[1, 3]"),
        (3.0, 5.0, "(3, 5]"),
        (5.0, 7.0, "(5, 7]"),
        (7.0, 10.0, "(7, 10]"),
    ]

    for left, right, label in bins:
        if left == 1.0:
            mask = (targets >= left) & (targets <= right)
        else:
            mask = (targets > left) & (targets <= right)

        idx = np.where(mask)[0]
        n = len(idx)
        if n == 0:
            print(f"{label}: 0 примеров")
            continue

        t_bin = targets[idx]
        p_bin = preds[idx]
        mae_bin = np.abs(p_bin - t_bin).mean()

        print(
            f"{label}: n={n:5d}, "
            f"mean target = {t_bin.mean():.4f}, "
            f"mean pred = {p_bin.mean():.4f}, "
            f"MAE = {mae_bin:.4f}"
        )

    print()


def main():
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Путь к модели:
    #  - если задан первым аргументом командной строки — используем его;
    #  - иначе берём последний по номеру чекпоинт ft.FINETUNE_BASENAME_XXX.pt.
    models_dir = ft.MODELS_DIR
    basename = ft.FINETUNE_BASENAME

    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            raise RuntimeError(f"Указанный файл модели не найден: {model_path}")
    else:
        model_path = get_latest_finetuned_model_path(models_dir, basename)

    print(f"Используем модель: {model_path}")

    # Загружаем модель
    model = load_model(model_path, device)

    # DataLoader'ы для val и test (берём df из finetune.py)
    val_loader = create_eval_loader(ft.val_df, batch_size=128)
    test_loader = create_eval_loader(ft.test_df, batch_size=128)

    # Собираем предсказания
    print("\n=== Сбор предсказаний на VAL ===")
    val_preds, val_targets = collect_predictions(model, val_loader, device)
    print("Готово.")

    print("\n=== Сбор предсказаний на TEST ===")
    test_preds, test_targets = collect_predictions(model, test_loader, device)
    print("Готово.\n")

    # Переводим в numpy для удобства
    val_preds_np = val_preds.numpy()
    val_targets_np = val_targets.numpy()
    test_preds_np = test_preds.numpy()
    test_targets_np = test_targets.numpy()

    # Общая сводная статистика
    print("========== VAL ==========")
    describe_array(val_targets_np, "VAL targets (mean_score по AVA)")
    describe_array(val_preds_np, "VAL preds   (mean_score модели)")
    describe_bins(val_preds_np, val_targets_np, "VAL")

    print("========== TEST ==========")
    describe_array(test_targets_np, "TEST targets (mean_score по AVA)")
    describe_array(test_preds_np, "TEST preds   (mean_score модели)")
    describe_bins(test_preds_np, test_targets_np, "TEST")


if __name__ == "__main__":
    main()
