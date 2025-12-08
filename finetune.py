from __future__ import annotations  # позволяет использовать аннотации типов, объявленных ниже по коду

import json
import math
import time
import random
import copy
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torch.amp import autocast, GradScaler  # новый API AMP: torch.amp.autocast / GradScaler

# ------------------------------------------------------------------------------
# ОТКЛЮЧЕНИЕ ШУМНЫХ ПРЕДУПРЕЖДЕНИЙ
# ------------------------------------------------------------------------------

# Некоторые TIFF-файлы в AVA могут быть "битые" или обрезанные, PIL будет
# ругаться предупреждениями "Truncated File Read". Они не критичны и засоряют лог.
warnings.filterwarnings(
    "ignore",
    message="Truncated File Read",
    category=UserWarning,
    module="PIL.TiffImagePlugin",
)

# ------------------------------------------------------------------------------
# КОНФИГУРАЦИЯ ПУТЕЙ И БАЗОВЫХ ГИПЕРПАРАМЕТРОВ
# ------------------------------------------------------------------------------

# Корневая директория проекта (относительно запуска скрипта).
# Предполагается, что структура:
#   .
#   ├─ data/
#   │   └─ AVA/
#   └─ models/
PROJECT_ROOT = Path(".")

# Папка с данными
DATA_ROOT = PROJECT_ROOT / "data"

# Папка с AVA-датасетом (изображения + разметка + списки train/val/test)
AVA_ROOT = DATA_ROOT / "AVA"

# Основной файл разметки AVA:
#   каждая строка соответствует одному изображению,
#   содержит image_id и гистограмму голосов (n1..n10).
AVA_TXT = AVA_ROOT / "AVA.txt"

# Файлы со списками идентификаторов изображений для train / val / test.
# В каждом файле по одному image_id на строку.
TRAIN_LS_LIST = AVA_ROOT / "aesthetics_image_lists" / "generic_ls_train.jpgl"
VAL_SS_LIST = AVA_ROOT / "aesthetics_image_lists" / "generic_ss_train.jpgl"
TEST_LIST = AVA_ROOT / "aesthetics_image_lists" / "generic_test.jpgl"

# Папка, где лежат JPEG-изображения AVA: <image_id>.jpg
IMAGES_DIR = AVA_ROOT / "images"

# Папка и имена моделей
MODELS_DIR = PROJECT_ROOT / "models"

# Имя уже обученной модели (старой, базовой) — если есть, её backbone
# можно использовать как инициализацию.
OLD_MODEL_NAME = "resnet50_ava_regression.pt"

# Базовое имя для новых чекпоинтов fine-tuning'а.
FINETUNE_BASENAME = "resnet50_ava_regression_finetune"

# Основные гиперпараметры обучения
BATCH_SIZE = 96       # размер батча (при AMP и мощной GPU можно пробовать выше)
NUM_EPOCHS = 50       # количество эпох fine-tuning'а

# Целевой learning rate для полного fine-tune ResNet-50 с AdamW.
LR = 2e-4
WEIGHT_DECAY = 2e-4   # L2-регуляризация в AdamW (weight decay)

SEED = 42             # фиксированный seed для частичной воспроизводимости

# Размер входного изображения для ResNet-50 после аугментаций/препроцессинга.
# ResNet ожидает квадратный вход (обычно 224), но мы используем 448 для лучшего качества.
IMG_SIZE = 448

# Warmup для learning rate — первые WARMUP_EPOCHS эпох постепенно увеличиваем LR
# от WARMUP_START_FACTOR * LR до полного LR.
WARMUP_EPOCHS = 2
WARMUP_START_FACTOR = 0.1  # первый шаг: 0.1 * LR, затем линейно до 1.0 * LR

# Параметры загрузки данных (должны быть адаптированы под CPU / систему):
NUM_WORKERS_TRAIN = 12   # число процессов-воркеров для train DataLoader
NUM_WORKERS_EVAL = 8     # число воркеров для val/test DataLoader
PREFETCH_FACTOR = 4      # каждый worker подготавливает несколько батчей наперёд

# ------------------------------------------------------------------------------
# ФИКСАЦИЯ СЛУЧАЙНОСТИ
# ------------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """
    Фиксирует seed в основных библиотеках (Python, NumPy, PyTorch),
    чтобы уменьшить разброс результатов между запусками.

    Важно: полная детерминированность *не гарантируется* из-за особенностей
    алгоритмов cuDNN, особенно при включённом benchmark-режиме.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Для всех доступных GPU
        torch.cuda.manual_seed_all(seed)

    # Ниже — настройки cuDNN. Мы сознательно выбираем скорость, а не строгую
    # детерминированность:
    torch.backends.cudnn.deterministic = False  # разрешаем недетерминированные алгоритмы
    torch.backends.cudnn.benchmark = True       # cuDNN подбирает самые быстрые варианты

# ------------------------------------------------------------------------------
# КОНФИГУРАЦИЯ ЛОССА И "ХВОСТОВ"
# ------------------------------------------------------------------------------

# Мы используем комплексный лосс, состоящий из трёх компонент:
# 1) KL-дивергенция между предсказанным распределением голосов и истинным.
# 2) MSE по mean_score (математическое ожидание оценки).
# 3) Ординационный лосс (BCE по порогам "score > k", k=1..9).
KL_WEIGHT = 1.0   # вес KL-компоненты
MSE_WEIGHT = 1.0  # вес MSE по средним оценкам
ORD_WEIGHT = 1.0  # вес ординационного лосса

# Мы уделяем особое внимание "хвостам" распределения mean_score:
# - низкие оценки (низкое эстетическое качество)
# - высокие оценки (очень качественные фото)
TAIL_LOW_THRESHOLD = 3.0   # порог для "низкого хвоста": mean_score <= 3.0
TAIL_HIGH_THRESHOLD = 7.0  # порог для "высокого хвоста": mean_score >= 7.0

# При построении sample_weight усиливаем хвосты, умножая их вес на этот фактор.
TAIL_WEIGHT_FACTOR = 4.0

# Ограничиваем итоговые sample_weight, чтобы не было экстремальных значений.
MIN_WEIGHT_MULT = 0.5   # нижняя граница веса
MAX_WEIGHT_MULT = 3.0   # верхняя граница веса

# Дополнительное усиление хвостов прямо внутри лосса:
# для примеров из хвостов итоговый лосс умножается на TAIL_LOSS_MULT.
TAIL_LOSS_MULT = 2.0

# ------------------------------------------------------------------------------
# ЧТЕНИЕ AVA.TXT И ПОДГОТОВКА ТАБЛИЦ
# ------------------------------------------------------------------------------

def read_ava_txt(path: Path) -> pd.DataFrame:
    """
    Читает агрегированный файл AVA.txt (по одному изображению на строку).

    Формат строки (упрощённо):
        <...> image_id n1 n2 ... n10 <...>

    В коде используется диапазон столбцов 1..11:
        - столбец 1:  image_id
        - столбцы 2–11: n1..n10, где nk — количество голосов за оценку k.

    Возвращает DataFrame со столбцами:
        image_id, n1, ..., n10, n_total, mean_score

    Здесь:
      n_total = sum_{k=1..10} nk
      mean_score = (sum_{k=1..10} k * nk) / n_total
    """
    # Имена колонок: image_id, n1, ..., n10
    cols = ["image_id"] + [f"n{k}" for k in range(1, 11)]

    # Читаем нужный диапазон колонок из файла:
    # usecols=range(1, 12) означает, что мы берём столбцы с индексами 1..11
    # (0-й столбец файла игнорируется).
    df = pd.read_csv(
        path,
        sep=" ",          # пробел в качестве разделителя
        header=None,      # в файле нет заголовка
        usecols=range(1, 12),
        names=cols,
    )

    vote_cols = [f"n{k}" for k in range(1, 11)]

    # Общее число голосов по изображению:
    # n_total = n1 + n2 + ... + n10
    # clip(lower=1) защищает от деления на 0.
    df["n_total"] = df[vote_cols].sum(axis=1).clip(lower=1)

    # Вычисляем mean_score как математическое ожидание оценки:
    #   mean_score = (1*n1 + 2*n2 + ... + 10*n10) / n_total
    weights = pd.Series(range(1, 11), index=vote_cols)
    df["mean_score"] = (df[vote_cols] * weights).sum(axis=1) / df["n_total"]

    # Приводим image_id к строковому типу, чтобы удобнее мапить на имена файлов .jpg
    df["image_id"] = df["image_id"].astype(str)

    # Возвращаем только нужные столбцы в удобном порядке
    return df[["image_id"] + vote_cols + ["n_total", "mean_score"]]


def read_id_list(path: Path) -> pd.DataFrame:
    """
    Читает список идентификаторов изображений из файлов *.jpgl,
    где в каждой строке записан один image_id.

    Пример строки:
        12345

    Возвращает DataFrame с одним столбцом:
        image_id (строковый).
    """
    df = pd.read_csv(path, header=None, names=["image_id"])
    df["image_id"] = df["image_id"].astype(str)
    return df

# Читаем полный AVA.txt
ava = read_ava_txt(AVA_TXT)

# Читаем списки идентификаторов для train / val / test
train_ids = read_id_list(TRAIN_LS_LIST)
val_ids = read_id_list(VAL_SS_LIST)
test_ids = read_id_list(TEST_LIST)

# Отфильтровываем общий DataFrame ava по id-шникам из соответствующих списков.
# Таким образом, train_df, val_df и test_df содержат только те строки,
# которые принадлежат соответствующим подвыборкам.
train_df = ava[ava["image_id"].isin(train_ids["image_id"])].copy()
val_df = ava[ava["image_id"].isin(val_ids["image_id"])].copy()
test_df = ava[ava["image_id"].isin(test_ids["image_id"])].copy()

# ------------------------------------------------------------------------------
# ПОСТРОЕНИЕ sample_weight: НАДЁЖНОСТЬ РАЗМЕТКИ + ХВОСТЫ
# ------------------------------------------------------------------------------

# Среднее количество голосов по train — используется как базовая шкала.
mean_votes_train = train_df["n_total"].mean()

# Базовый вес примера пропорционален числу голосов:
# base_weight_i = n_total_i / mean_votes_train
# Чем больше голосов, тем надёжнее разметка → выше вес.
base_weight = train_df["n_total"] / mean_votes_train

# Выделяем хвосты по mean_score:
tail_mask_low = train_df["mean_score"] <= TAIL_LOW_THRESHOLD
tail_mask_high = train_df["mean_score"] >= TAIL_HIGH_THRESHOLD
tail_mask = tail_mask_low | tail_mask_high

# Начинаем с множителя 1.0 для всех примеров
tail_boost = pd.Series(1.0, index=train_df.index)
# Для хвостов увеличиваем вес в TAIL_WEIGHT_FACTOR раз
tail_boost[tail_mask] = TAIL_WEIGHT_FACTOR

# Итоговый вес train-примера:
# sample_weight_i = base_weight_i * tail_boost_i
sample_weight_train = base_weight * tail_boost

# Нормируем веса так, чтобы средний вес был 1.0
mean_weight = sample_weight_train.mean()
sample_weight_train = sample_weight_train / mean_weight

# Ограничиваем веса в диапазоне [MIN_WEIGHT_MULT, MAX_WEIGHT_MULT],
# чтобы не было слишком экстремальных значений.
sample_weight_train = sample_weight_train.clip(lower=MIN_WEIGHT_MULT, upper=MAX_WEIGHT_MULT)

# Сохраняем sample_weight в train_df для последующего использования в датасете
train_df["sample_weight"] = sample_weight_train


def compute_eval_weights(df: pd.DataFrame) -> pd.Series:
    """
    Строит sample_weight для валидации и теста по той же логике, что и для train.

    Идея: при оценке мы тоже можем учитывать "надёжность" разметки и хвосты,
    например при расчёте взвешенных метрик. Здесь же веса будут храниться
    в df["sample_weight"], хотя обычно валидацию считают без самплирования.
    """
    # Аналог base_weight, но для другого df
    base_w = df["n_total"] / mean_votes_train

    # Определяем хвосты по mean_score
    tail_mask_low = df["mean_score"] <= TAIL_LOW_THRESHOLD
    tail_mask_high = df["mean_score"] >= TAIL_HIGH_THRESHOLD
    tail_mask = tail_mask_low | tail_mask_high

    # Аналог tail_boost для val/test
    boost = pd.Series(1.0, index=df.index)
    boost[tail_mask] = TAIL_WEIGHT_FACTOR

    # Итоговый вес
    weights = base_w * boost

    # Нормируем веса так, чтобы среднее было 1.0
    weights = weights / weights.mean()

    # Ограничиваем диапазон
    weights = weights.clip(lower=MIN_WEIGHT_MULT, upper=MAX_WEIGHT_MULT)
    return weights

# Считаем sample_weight для val и test
val_df["sample_weight"] = compute_eval_weights(val_df)
test_df["sample_weight"] = compute_eval_weights(test_df)

# ------------------------------------------------------------------------------
# ПОСТРОЕНИЕ ПУТЕЙ К КАРТИНКАМ
# ------------------------------------------------------------------------------

def make_path(image_id: str) -> Path:
    """
    По идентификатору AVA (строка) строит путь к файлу изображения .jpg.

    Пример:
      image_id = "12345"
      → IMAGES_DIR / "12345.jpg"
    """
    return IMAGES_DIR / f"{image_id}.jpg"


# Для каждого датафрейма добавляем столбец image_path,
# чтобы дальше датасет мог быстро получить путь к JPEG.
for df in (train_df, val_df, test_df):
    df["image_path"] = df["image_id"].apply(make_path)

# ------------------------------------------------------------------------------
# ТРАНСФОРМАЦИИ ИЗОБРАЖЕНИЙ
# ------------------------------------------------------------------------------

def get_train_transform():
    """
    Трансформации для обучения (аугментации).

    Здесь мы:
      1) Случайно кадрируем часть изображения (RandomResizedCrop) с масштабом
         от 0.8 до 1.0 исходного размера и приводим к IMG_SIZE x IMG_SIZE.
      2) Случайно отражаем по горизонтали (flip).
      3) Слегка варьируем яркость, контраст, насыщенность и оттенок (ColorJitter).
      4) Иногда переводим в оттенки серого (RandomGrayscale).
      5) Конвертируем в тензор (ToTensor).
      6) Нормируем по статистикам ImageNet (Normalize).
    """
    return T.Compose(
        [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            ),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # стандартные значения ImageNet
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transform():
    """
    Трансформации для валидации и теста.

    В отличие от train-аугментаций, здесь всё детерминированно:
      1) Масштабируем изображение так, чтобы меньшая сторона была чуть больше (коэффициент 1.143),
      2) Делаем центрированный кроп до IMG_SIZE x IMG_SIZE.
      3) Конвертируем в тензор.
      4) Нормируем по тем же статистикам ImageNet.
    """
    return T.Compose(
        [
            T.Resize(int(IMG_SIZE * 1.143)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

# ------------------------------------------------------------------------------
# DATASET: ИЗОБРАЖЕНИЕ + РАСПРЕДЕЛЕНИЕ ГОЛОСОВ + MEAN_SCORE + ВЕС
# ------------------------------------------------------------------------------

class AVADistributionDataset(Dataset):
    """
    Кастомный Dataset для AVA.

    Для каждого примера он возвращает:
      - изображение (тензор [3 x H x W]),
      - распределение голосов p_k для оценок 1..10 (тензор [10]),
      - mean_score (скаляр),
      - sample_weight (скаляр).

    Здесь p_k — это нормированное количество голосов:
        p_k = n_k / n_total,
    где n_k — количество голосов за оценку k, n_total — сумма по всем k.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        # Сохраняем DataFrame и трансформацию.
        # reset_index(drop=True) — чтобы индексы шли 0..N-1 подряд.
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        # Длина датасета — число строк в DataFrame
        return len(self.df)

    def _load_image_safe(self, path: Path) -> Image.Image | None:
        """
        Безопасная загрузка изображения с обработкой ошибок чтения.

        Если при открытии файла возникает исключение (например,
        файл битый), метод возвращает None.
        """
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            print(f"Warning: failed to load image {path}: {e}")
            return None

    def __getitem__(self, idx: int):
        # Достаём строку с индексом idx
        row = self.df.iloc[idx]
        path = row["image_path"]

        # Делаем до 3 попыток "сдвинуться" по датасету вперёд, если файл битый.
        for _ in range(3):
            img = self._load_image_safe(path)
            if img is not None:
                break
            # Если не удалось — переключаемся на следующий пример по кругу
            idx = (idx + 1) % len(self.df)
            row = self.df.iloc[idx]
            path = row["image_path"]
        else:
            # Если не получилось даже за 3 попытки — создаём чёрное "заглушечное" изображение,
            # чтобы не ломать обучение.
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
            print(f"Warning: using black dummy image for {path}")

        # Применяем трансформацию (аугментацию/препроцессинг), если она есть.
        if self.transform is not None:
            img = self.transform(img)

        # Считываем сырые количества голосов n1..n10 из строки и делаем тензор.
        counts = torch.tensor(
            [row[f"n{k}"] for k in range(1, 11)],
            dtype=torch.float32,
        )

        # total — общее число голосов, но не меньше 1 (защита от деления на 0).
        total = counts.sum().clamp(min=1.0)

        # dist — нормированное распределение голосов:
        #   dist_k = n_k / total.
        dist = counts / total

        # mean_score и sample_weight — скаляры из DataFrame.
        mean_score = torch.tensor(row["mean_score"], dtype=torch.float32)
        sample_weight = torch.tensor(row["sample_weight"], dtype=torch.float32)

        # Возвращаем кортеж, который далее попадает в DataLoader
        return img, dist, mean_score, sample_weight

# ------------------------------------------------------------------------------
# МОДЕЛЬ: RESNET-50 С ДВУМЯ ГОЛОВАМИ
# ------------------------------------------------------------------------------

class ResNet50AvaMultiHead(nn.Module):
    """
    Модель на базе ResNet-50 для задачи эстетической оценки AVA.

    Состоит из:
      - backbone: ResNet-50 без финального полносвязного слоя,
      - fc_dist: голова, предсказывающая логиты распределения по 10 оценкам (1..10),
      - fc_ord: ординационная голова (9 логитов, соответствующих порогам score > k, k=1..9).

    Идея:
      1) fc_dist даёт распределение голосов, которое сравнивается с истинным
         через KL-дивергенцию.
      2) Из распределения вычисляем mean_score и используем MSE.
      3) fc_ord моделирует вероятность того, что оценка строго больше k.
         Это даёт ординационный лосс, учитывающий порядок оценок.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Загружаем ResNet-50:
        #   pretrained=True → веса, предобученные на ImageNet.
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Число признаков на выходе из последнего слоя (перед fc) — обычно 2048.
        in_features = backbone.fc.in_features

        # Заменяем последний слой (fc) на Identity, чтобы backbone возвращал
        # только вектор признаков, а не логиты по классам ImageNet.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Голова для распределения оценок: линейный слой → 10 логитов.
        self.fc_dist = nn.Linear(in_features, 10)

        # Ординационная голова: линейный слой → 9 логитов (для порогов 1..9).
        self.fc_ord = nn.Linear(in_features, 9)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Сначала прогоняем изображения через backbone (ResNet без финальной fc).
        features = self.backbone(x)

        # Логиты распределения для оценок 1..10
        logits_dist = self.fc_dist(features)

        # Логиты для ординационных порогов score > k, k=1..9
        logits_ord = self.fc_ord(features)

        # Возвращаем оба набора логитов
        return logits_dist, logits_ord

# ------------------------------------------------------------------------------
# МЕТРИКИ РЕГРЕССИИ ПО MEAN_SCORE
# ------------------------------------------------------------------------------

@dataclass
class RegressionMetrics:
    """
    Структура для хранения метрик регрессии по mean_score:

    - mse, rmse, mae — по всей выборке,
    - mse_low, rmse_low, mae_low — по низкому хвосту (mean_score <= low_threshold),
    - mse_high, rmse_high, mae_high — по высокому хвосту (mean_score >= high_threshold).
    """
    mse: float
    rmse: float
    mae: float

    mse_low: float
    rmse_low: float
    mae_low: float

    mse_high: float
    rmse_high: float
    mae_high: float


def compute_regression_metrics(
        preds: torch.Tensor,      # предсказанные mean_score
        targets: torch.Tensor,    # истинные mean_score
        target_means: torch.Tensor,  # те же targets, но явно (для масок хвостов)
        low_threshold: float,
        high_threshold: float,
) -> RegressionMetrics:
    """
    Вычисляет MSE / RMSE / MAE по всей выборке и отдельно по хвостам.

    Пусть:
      e_i = preds_i - targets_i — ошибка для примера i.

    Тогда:
      MSE = mean( e_i^2 )
      RMSE = sqrt(MSE)
      MAE = mean( |e_i| )

    Для хвостов считаем те же величины, но только по подмножествам:
      - low:   target_means_i <= low_threshold
      - high:  target_means_i >= high_threshold
    """
    errors = preds - targets

    # Метрики по всей выборке
    mse = (errors ** 2).mean().item()
    rmse = math.sqrt(mse)
    mae = errors.abs().mean().item()

    # Маски хвостов
    low_mask = target_means <= low_threshold
    high_mask = target_means >= high_threshold

    def safe_stats(mask: torch.Tensor) -> Tuple[float, float, float]:
        """
        Вспомогательная функция: считает MSE/RMSE/MAE для поднабора данных.

        Если поднабор пустой (mask.sum() == 0), возвращает NaN.
        """
        if mask.sum().item() == 0:
            return float("nan"), float("nan"), float("nan")
        e = errors[mask]
        mse_ = (e ** 2).mean().item()
        rmse_ = math.sqrt(mse_)
        mae_ = e.abs().mean().item()
        return mse_, rmse_, mae_

    # Метрики по хвостам
    mse_low, rmse_low, mae_low = safe_stats(low_mask)
    mse_high, rmse_high, mae_high = safe_stats(high_mask)

    return RegressionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        mse_low=mse_low,
        rmse_low=rmse_low,
        mae_low=mae_low,
        mse_high=mse_high,
        rmse_high=rmse_high,
        mae_high=mae_high,
    )

# ------------------------------------------------------------------------------
# ОБУЧЕНИЕ ОДНОЙ ЭПОХИ
# ------------------------------------------------------------------------------

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler | None = None,
) -> float:
    """
    Обучает модель одну эпоху на train_loader.

    Внутри реализован следующий лосс на каждый пример:
        loss = KL_WEIGHT * KL(p_true || p_pred)
             + MSE_WEIGHT * (mean_pred - mean_true)^2
             + ORD_WEIGHT * ord_loss

    где:
      - p_true — истинное распределение голосов (dist из датасета),
      - p_pred — предсказанное моделью распределение (через softmax),
      - mean_pred — ожидание оценки под p_pred,
      - mean_true — mean_score из датасета,
      - ord_loss — ординационный BCE по порогам "score > k", k=1..9.

    Далее:
      - Для хвостов (mean_true <= TAIL_LOW_THRESHOLD или >= TAIL_HIGH_THRESHOLD)
        лосс умножается на TAIL_LOSS_MULT.
      - После этого лосс взвешивается sample_weight и усредняется по батчу.
    """
    model.train()  # переводим модель в режим обучения (включает dropout, BN и т.п.)

    total_loss = 0.0     # накопитель для суммарного лосса (с учётом веса батча)
    total_weight = 0.0   # накопитель для суммарного веса примеров в батчах

    # Определяем, используем ли AMP (Automatic Mixed Precision).
    use_amp = (
            scaler is not None
            and scaler.is_enabled()
            and device.type == "cuda"
    )

    # Основной цикл по батчам
    for images, target_dists, target_means, sample_weights in loader:
        # Переносим всё на GPU (если есть), максимально эффективно.
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,  # формат памяти, оптимальный для свёрток
        )
        target_dists = target_dists.to(device, non_blocking=True)
        target_means = target_means.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)

        # Обнуляем градиенты перед шагом оптимизатора
        optimizer.zero_grad()

        # Блок AMP: часть операций будет считаться в float16 для ускорения.
        with autocast(
                "cuda",
                enabled=use_amp,
                dtype=torch.float16,
        ):
            # Прогоняем батч через модель → получаем:
            #   logits_dist: [B, 10] — логиты по оценкам 1..10
            #   logits_ord:  [B, 9]  — логиты по порогам score > k
            logits_dist, logits_ord = model(images)

            # Переводим логиты распределения в лог-вероятности:
            #   log_probs = log_softmax(logits_dist)
            log_probs = F.log_softmax(logits_dist, dim=1)

            # Предсказанное распределение:
            #   p_pred_k = exp(log_probs_k)
            pred_probs = log_probs.exp()

            # Вектор оценок [1, 2, ..., 10], приведённый к размерности [1, 10]
            ks = torch.arange(1, 11, device=device, dtype=pred_probs.dtype).view(1, -1)

            # Предсказанный mean_score как ожидание:
            #   mean_pred = sum_k k * p_pred_k
            pred_means = (pred_probs * ks).sum(dim=1)

            # -------------------- KL-компонента лосса --------------------
            # F.kl_div ожидает на вход log_probs (логарифм предсказанного распределения)
            # и target-дистрибуцию (target_dists).
            #
            # reduction="none" возвращает тензор размера [B, 10].
            # Мы суммируем по измерению оценок (dim=1), получая KL для каждого примера.
            kl_per_sample = F.kl_div(
                log_probs,
                target_dists,
                reduction="none",
            ).sum(dim=1)

            # -------------------- MSE по mean_score --------------------
            # mse_i = (mean_pred_i - mean_true_i)^2
            mse_per_sample = (pred_means - target_means) ** 2

            # -------------------- Ординационный лосс --------------------
            # Для ординационного подхода нам нужно моделировать вероятности:
            #   P(score > k) для k=1..9.
            #
            # Из истинного распределения p_true можно получить CDF:
            #   P(score <= k) = sum_{j <= k} p_true_j
            cdf_le = torch.cumsum(target_dists, dim=1)    # [B, 10]
            p_le_k = cdf_le[:, :-1]                       # [B, 9], k=1..9
            p_gt_k = 1.0 - p_le_k                         # P(score > k) = 1 - P(score <= k)

            # logits_ord — логиты для "score > k".
            # Используем бинарную кросс-энтропию (BCEWithLogits) по каждому порогу:
            ord_loss_all = F.binary_cross_entropy_with_logits(
                logits_ord,
                p_gt_k,
                reduction="none",
            )
            # Усредняем по порогам → получаем ординационный лосс на каждый пример.
            ord_loss_per_sample = ord_loss_all.mean(dim=1)

            # -------------------- Составляем общий лосс на пример --------------------
            # loss_i = KL_WEIGHT * KL_i + MSE_WEIGHT * MSE_i + ORD_WEIGHT * ORD_i
            loss_per_sample = (
                    KL_WEIGHT * kl_per_sample
                    + MSE_WEIGHT * mse_per_sample
                    + ORD_WEIGHT * ord_loss_per_sample
            )

            # Усиление хвостов: если target_mean_i в хвостах,
            # умножаем лосс на TAIL_LOSS_MULT.
            tail_mask = (target_means <= TAIL_LOW_THRESHOLD) | (
                    target_means >= TAIL_HIGH_THRESHOLD
            )
            tail_factor = torch.ones_like(loss_per_sample)
            tail_factor[tail_mask] = TAIL_LOSS_MULT
            loss_per_sample = loss_per_sample * tail_factor

            # -------------------- Взвешивание sample_weight --------------------
            # Формула итогового лосса по батчу:
            #   weighted_loss = sum_i (loss_i * w_i) / sum_i w_i
            weighted_loss = (
                    (loss_per_sample * sample_weights).sum() / sample_weights.sum()
            )

        # Шаг оптимизации с учётом AMP
        if use_amp:
            scaler.scale(weighted_loss).backward()  # считаем градиенты в масштабированном виде
            scaler.step(optimizer)                  # делаем шаг оптимизатора
            scaler.update()                         # обновляем масштаб
        else:
            weighted_loss.backward()
            optimizer.step()

        # Считаем "массу" батча (сумму sample_weight) и обновляем накопители.
        batch_weight = sample_weights.sum().item()
        total_loss += weighted_loss.detach().item() * batch_weight
        total_weight += batch_weight

    # Средний лосс по эпохе с учётом sample_weight.
    mean_loss = total_loss / max(total_weight, 1.0)
    return mean_loss

# ------------------------------------------------------------------------------
# ОЦЕНКА МОДЕЛИ
# ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> RegressionMetrics:
    """
    Оценка модели по mean_score на заданном DataLoader.

    Важно: во время оценки:
      - не считаются градиенты (torch.no_grad),
      - модель переключается в режим eval() (замораживаются batchnorm/dropout),
      - используется только регрессионная часть: мы получаем mean_pred и
        сравниваем его с mean_true через RegressionMetrics.
    """
    model.eval()  # режим оценки

    all_pred_means = []   # список, в который будем собирать предсказанные mean_score
    all_target_means = [] # список для истинных mean_score

    use_amp = (device.type == "cuda")

    for images, target_dists, target_means, sample_weights in loader:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        target_means = target_means.to(device, non_blocking=True)

        # AMP можно использовать и во время оценки для ускорения.
        with autocast(
                "cuda",
                enabled=use_amp,
                dtype=torch.float16,
        ):
            logits_dist, logits_ord = model(images)

            # Как и в обучении: получаем предсказанное распределение
            # и из него вычисляем mean_pred.
            log_probs = F.log_softmax(logits_dist, dim=1)
            pred_probs = log_probs.exp()

            ks = torch.arange(1, 11, device=device, dtype=pred_probs.dtype).view(1, -1)
            pred_means = (pred_probs * ks).sum(dim=1)

        # Переносим на CPU и добавляем в списки
        all_pred_means.append(pred_means.cpu())
        all_target_means.append(target_means.cpu())

    # Склеиваем списки тензоров вдоль нулевой оси → получаем большие векторы.
    all_pred_means = torch.cat(all_pred_means, dim=0)
    all_target_means = torch.cat(all_target_means, dim=0)

    # Считаем метрики регрессии, включая хвосты.
    metrics = compute_regression_metrics(
        preds=all_pred_means,
        targets=all_target_means,
        target_means=all_target_means,
        low_threshold=TAIL_LOW_THRESHOLD,
        high_threshold=TAIL_HIGH_THRESHOLD,
    )

    return metrics

# ------------------------------------------------------------------------------
# DATALOADER'Ы И ВЗВЕШЕННОЕ СЕМПЛИРОВАНИЕ
# ------------------------------------------------------------------------------

def create_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт три DataLoader'а: train / val / test.

    Особенности:
      - train_loader:
          * использует AVADistributionDataset с train_df,
          * применяет аугментации (get_train_transform),
          * использует WeightedRandomSampler по sample_weight (таргетная оптимизация хвостов).

      - val_loader и test_loader:
          * AVADistributionDataset с val_df / test_df,
          * детерминированные трансформации (get_eval_transform),
          * обычная sequential-подача (shuffle=False).

      - Увеличено число воркеров, включены pin_memory, persistent_workers,
        prefetch_factor — всё для максимальной загрузки GPU.
    """
    # Создаём датасеты
    train_ds = AVADistributionDataset(train_df, transform=get_train_transform())
    val_ds = AVADistributionDataset(val_df, transform=get_eval_transform())
    test_ds = AVADistributionDataset(test_df, transform=get_eval_transform())

    # Преобразуем sample_weight из train_df в тензор для WeightedRandomSampler
    train_weights = torch.tensor(
        train_df["sample_weight"].values,
        dtype=torch.float32,
    )

    # WeightedRandomSampler с вероятностью, пропорциональной train_weights.
    # num_samples=len(train_weights) означает, что за одну "эпоху" мы берем
    # столько же сэмплов, сколько и строк в train_df (с повторениями).
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,  # сэмплирование с возвращением
    )

    # DataLoader для train:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,     # вместо shuffle
        shuffle=False,             # shuffle выключен, т.к. уже есть sampler
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True,           # ускоряет перенос данных на GPU
        persistent_workers=True,   # воркеры не перезапускаются каждый epoch
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=True,            # отбрасываем последний неполный батч (если есть)
    )

    # DataLoader для val:
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,             # валидация в фиксированном порядке
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    # DataLoader для test:
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    return train_loader, val_loader, test_loader

# ------------------------------------------------------------------------------
# АВТОНОМЕРАЦИЯ ИМЁН ДЛЯ НОВЫХ МОДЕЛЕЙ
# ------------------------------------------------------------------------------

def get_next_finetune_paths() -> Tuple[Path, Path]:
    """
    Определяет следующее свободное имя для чекпоинта и файла с метриками.

    Логика:
      - в MODELS_DIR ищем все файлы вида "FINETUNE_BASENAME_*.pt",
      - вытаскиваем максимальный индекс N,
      - новый чекпоинт будет называться FINETUNE_BASENAME_{N+1:03d}.pt,
      - файл метрик — FINETUNE_BASENAME_{N+1:03d}_metrics.json.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    # Ищем существующие файлы чекпоинтов
    existing = list(MODELS_DIR.glob(f"{FINETUNE_BASENAME}_*.pt"))

    max_idx = 0
    for p in existing:
        stem = p.stem  # имя файла без расширения
        # Ожидаем формат: "<FINETUNE_BASENAME>_<число>"
        suffix = stem.replace(FINETUNE_BASENAME + "_", "")
        try:
            idx = int(suffix)
            max_idx = max(max_idx, idx)
        except ValueError:
            # Если после подчёркивания не число — игнорируем
            continue

    # Следующий индекс
    next_idx = max_idx + 1

    # Формируем имена файлов
    name = f"{FINETUNE_BASENAME}_{next_idx:03d}.pt"
    metrics_name = f"{FINETUNE_BASENAME}_{next_idx:03d}_metrics.json"

    return MODELS_DIR / name, MODELS_DIR / metrics_name

# ------------------------------------------------------------------------------
# ОСНОВНОЙ СКРИПТ ОБУЧЕНИЯ И ОЦЕНКИ
# ------------------------------------------------------------------------------

def main() -> None:
    """
    Основная функция, которая запускается при выполнении скрипта.

    Последовательность действий:
      1) Фиксируем seed и создаём папку models.
      2) Определяем устройство (CPU / GPU).
      3) Создаём DataLoader'ы для train / val / test.
      4) Инициализируем модель:
           - либо загружаем backbone из старого чекпоинта (если есть),
           - либо используем torchvision pretrained ResNet-50.
      5) Оцениваем "старую" модель на val и test (до fine-tune).
      6) Настраиваем оптимизатор AdamW, GradScaler (AMP) и scheduler.
      7) Запускаем цикл fine-tuning на NUM_EPOCHS эпох:
           - train_one_epoch,
           - evaluate_model на val,
           - шаг scheduler,
           - обновление best_state_dict по лучшему val MSE.
      8) Сохраняем лучший чекпоинт и метрики до/после обучения.
      9) Выводим сравнение метрик "до" и "после".
    """
    # Фиксируем seed для уменьшения разброса результатов
    set_seed(SEED)

    # Убеждаемся, что папка для моделей существует
    MODELS_DIR.mkdir(exist_ok=True)

    # Определяем устройство: GPU, если доступен, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Создаём DataLoader'ы
    train_loader, val_loader, test_loader = create_dataloaders(BATCH_SIZE)

    # Путь к "старой" модели
    old_model_path = MODELS_DIR / OLD_MODEL_NAME

    # Инициализация модели и загрузка backbone-весов (если есть старый чекпоинт)
    if old_model_path.exists():
        print(f"Loading backbone weights from {old_model_path}")
        # Создаём модель без предобученных ImageNet-весов (pretrained=False),
        # так как веса уже будут загружены из чекпоинта.
        model = ResNet50AvaMultiHead(pretrained=False).to(device)
        # Используем формат памяти channels_last для ускорения свёрток на GPU
        model = model.to(memory_format=torch.channels_last)

        # Загружаем содержимое файла с моделью
        state: Any = torch.load(old_model_path, map_location=device)

        # Если чекпоинт сохранён как {"state_dict": ...}, извлекаем state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Если ключи не содержат префикс "backbone.", добавляем его.
        # Это нужно, если старая модель была только backbone-ом ResNet-50.
        if isinstance(state, dict) and not any(k.startswith("backbone.") for k in state.keys()):
            state = {f"backbone.{k}": v for k, v in state.items()}

        if isinstance(state, dict):
            # Удаляем ключи, относящиеся к старой fc (backbone.fc), т.к. у нас
            # другая архитектура голов.
            state = {k: v for k, v in state.items() if not k.startswith("backbone.fc")}
            # Загружаем состояние в модель с допуском несовпадения ключей
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"Warning: missing keys when loading checkpoint: {missing}")
            if unexpected:
                print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")
        else:
            # Если формат чекпоинта неожиданен, не падаем — просто сообщаем
            print("Warning: unexpected checkpoint format, initializing backbone from scratch.")
    else:
        # Если старого чекпоинта нет, берём ResNet-50 с предобученными весами ImageNet.
        print(
            f"Base model file not found: {old_model_path}, "
            "using torchvision pretrained ResNet-50."
        )
        model = ResNet50AvaMultiHead(pretrained=True).to(device)
        model = model.to(memory_format=torch.channels_last)

    # ------------------ Оценка "старой" модели до fine-tune ------------------
    print("\n=== Old model evaluation ===")
    old_val_metrics = evaluate_model(model, val_loader, device)
    old_test_metrics = evaluate_model(model, test_loader, device)

    # Выводим метрики на валидации (до обучения)
    print(
        f"Old VAL  -> MSE: {old_val_metrics.mse:.4f}, "
        f"RMSE: {old_val_metrics.rmse:.4f}, MAE: {old_val_metrics.mae:.4f}"
    )
    print(
        f"           low (<= {TAIL_LOW_THRESHOLD}): "
        f"MSE {old_val_metrics.mse_low:.4f}, "
        f"RMSE {old_val_metrics.rmse_low:.4f}, "
        f"MAE {old_val_metrics.mae_low:.4f}"
    )
    print(
        f"           high (>= {TAIL_HIGH_THRESHOLD}): "
        f"MSE {old_val_metrics.mse_high:.4f}, "
        f"RMSE {old_val_metrics.rmse_high:.4f}, "
        f"MAE {old_val_metrics.mae_high:.4f}"
    )

    # Метрики на тесте (до обучения)
    print(
        f"Old TEST -> MSE: {old_test_metrics.mse:.4f}, "
        f"RMSE: {old_test_metrics.rmse:.4f}, MAE: {old_test_metrics.mae:.4f}"
    )
    print(
        f"           low (<= {TAIL_LOW_THRESHOLD}): "
        f"MSE {old_test_metrics.mse_low:.4f}, "
        f"RMSE {old_test_metrics.rmse_low:.4f}, "
        f"MAE {old_test_metrics.mae_low:.4f}"
    )
    print(
        f"           high (>= {TAIL_HIGH_THRESHOLD}): "
        f"MSE {old_test_metrics.mse_high:.4f}, "
        f"RMSE {old_test_metrics.rmse_high:.4f}, "
        f"MAE {old_test_metrics.mae_high:.4f}"
    )

    # ------------------ Настройка оптимизатора, GradScaler и scheduler'а ------------------

    # AdamW — адаптивный оптимизатор с weight decay (L2-регуляризация).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # GradScaler для AMP (новый API: torch.amp.GradScaler).
    scaler = GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    # ReduceLROnPlateau — scheduler, который уменьшает LR, если метрика (val MSE)
    # перестаёт улучшаться в течение нескольких эпох (patience).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        # verbose параметр больше не используется (чтобы избежать предупреждений).
    )

    # Лучшая достигнутая MSE на валидации и соответствующее состояние модели.
    best_val_mse = old_val_metrics.mse
    best_state_dict = copy.deepcopy(model.state_dict())

    # ------------------ Цикл fine-tuning'а ------------------
    print("\n=== Finetuning ===")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # ----------- Warmup для LR -----------

        if epoch <= WARMUP_EPOCHS:
            # Если warmup из одной эпохи — просто используем LR без прогрессии.
            if WARMUP_EPOCHS == 1:
                warmup_factor = 1.0
            else:
                # warmup_progress ∈ [0, 1] — доля прохождения warmup-интервала
                warmup_progress = (epoch - 1) / (WARMUP_EPOCHS - 1)
                # Линейно интерполируем между WARMUP_START_FACTOR*LR и LR
                warmup_factor = (
                        WARMUP_START_FACTOR
                        + (1.0 - WARMUP_START_FACTOR) * warmup_progress
                )
            current_lr = LR * warmup_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            # После окончания warmup используем LR, управляемый scheduler'ом
            current_lr = optimizer.param_groups[0]["lr"]

        # Один проход по train — обучение
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )

        # Оценка на валидации
        val_metrics = evaluate_model(model, val_loader, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"lr: {current_lr:.6f} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_MSE: {val_metrics.mse:.4f} | "
            f"val_RMSE: {val_metrics.rmse:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        # Сообщаем scheduler'у значение метрики, по которой он должен реагировать
        scheduler.step(val_metrics.mse)

        # Если текущая MSE на валидации лучше, чем предыдущий минимум —
        # обновляем "лучшее" состояние модели.
        if val_metrics.mse < best_val_mse:
            best_val_mse = val_metrics.mse
            best_state_dict = copy.deepcopy(model.state_dict())

    # ------------------ Сохранение лучшей модели и метрик ------------------

    new_model_path, metrics_path = get_next_finetune_paths()

    # Сохраняем state_dict лучшей модели
    torch.save(best_state_dict, new_model_path)
    print(f"\nSaved finetuned model to {new_model_path}")

    # Загружаем лучшие веса обратно в модель для финальной оценки
    model.load_state_dict(best_state_dict)

    # Метрики новой (дообученной) модели
    new_val_metrics = evaluate_model(model, val_loader, device)
    new_test_metrics = evaluate_model(model, test_loader, device)

    print("\n=== Before / after comparison ===")

    def show_metrics(name: str, old: RegressionMetrics, new: RegressionMetrics) -> None:
        """
        Удобная функция для красивого вывода метрик до/после обучения.
        """
        print(
            f"{name} ALL   -> "
            f"MSE: {old.mse:.4f} -> {new.mse:.4f}, "
            f"RMSE: {old.rmse:.4f} -> {new.rmse:.4f}, "
            f"MAE: {old.mae:.4f} -> {new.mae:.4f}"
        )
        print(
            f"{name} LOW   -> "
            f"MSE: {old.mse_low:.4f} -> {new.mse_low:.4f}, "
            f"RMSE: {old.rmse_low:.4f} -> {new.rmse_low:.4f}, "
            f"MAE: {old.mae_low:.4f} -> {new.mae_low:.4f}"
        )
        print(
            f"{name} HIGH  -> "
            f"MSE: {old.mse_high:.4f} -> {new.mse_high:.4f}, "
            f"RMSE: {old.rmse_high:.4f} -> {new.rmse_high:.4f}, "
            f"MAE: {old.mae_high:.4f} -> {new.mae_high:.4f}"
        )

    # Показываем сравнение "до" и "после" для val и test
    show_metrics("VAL ", old_val_metrics, new_val_metrics)
    show_metrics("TEST", old_test_metrics, new_test_metrics)

    # Формируем словарь с метриками и параметрами обучения для сохранения в JSON.
    metrics_payload: Dict[str, Dict] = {
        "old_val": asdict(old_val_metrics),
        "old_test": asdict(old_test_metrics),
        "new_val": asdict(new_val_metrics),
        "new_test": asdict(new_test_metrics),
        "train_params": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "tail_low_threshold": TAIL_LOW_THRESHOLD,
            "tail_high_threshold": TAIL_HIGH_THRESHOLD,
            "tail_weight_factor": TAIL_WEIGHT_FACTOR,
            "tail_loss_mult": TAIL_LOSS_MULT,
            "kl_weight": KL_WEIGHT,
            "mse_weight": MSE_WEIGHT,
            "ord_weight": ORD_WEIGHT,
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup_start_factor": WARMUP_START_FACTOR,
            "amp_enabled": (device.type == "cuda"),
            "num_workers_train": NUM_WORKERS_TRAIN,
            "num_workers_eval": NUM_WORKERS_EVAL,
            "prefetch_factor": PREFETCH_FACTOR,
        },
    }

    # Сохраняем JSON с метриками и параметрами
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics to {metrics_path}")


# Точка входа: если файл запускается как скрипт, вызываем main().
if __name__ == "__main__":
    main()
