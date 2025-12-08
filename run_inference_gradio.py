from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision import models
import gradio as gr

# ===== Настройки путей и устройства =====

PROJECT_ROOT = Path(".")
MODELS_DIR = PROJECT_ROOT / "models"
# сюда положен лучший чекпоинт из finetune.py
MODEL_PATH = MODELS_DIR / "resnet50_ava_regression_finetune_001.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Должно совпадать с IMG_SIZE в обучающем скрипте
IMG_SIZE = 448

# ===== Нормализация под человеческую шкалу 0–10 =====
# В этих пределах обычно живут "сырые" предсказания модели (1–10 шкала AVA).
# Лучше всего подобрать их по предсказаниям на валидейшене.
RAW_MIN = 2.74386 # 1.9599 + 40% примерно "плохая" картинка по модели
RAW_MAX = 7.42572   # 8.2508 - 10% примерно "очень хорошая" картинка по модели
# коэффициенты подобраны через скрипт raw_compute.py +- 40%

def normalize_to_human_scale(raw_score: float) -> float:
    """
    Линейно переводит исходный mean_score модели (примерно RAW_MIN..RAW_MAX)
    в диапазон 0..10 и обрезает за его пределами.

    raw_score ~ RAW_MIN  ->  ~0
    raw_score ~ RAW_MAX  -> ~10
    raw_score между ними -> 0..10 по линейной шкале.
    """
    if RAW_MAX <= RAW_MIN:
        # защитный вариант, если вдруг кто-то неправильно настроил константы
        return max(0.0, min(10.0, raw_score))

    x = (raw_score - RAW_MIN) / (RAW_MAX - RAW_MIN)  # 0..1 (в идеале)
    x = max(0.0, min(1.0, x))                        # клип в [0, 1]
    return x * 10.0


# ===== Модель (ResNet50AvaMultiHead, как в finetune.py) =====

class ResNet50AvaMultiHead(nn.Module):
    """
    ResNet-50 для задачи эстетики AVA с двумя головами.

    Голова 1 (fc_dist): распределение по 10 оценкам (1..10).
    Голова 2 (fc_ord): ординационная (score > k, k=1..9).
    Для инференса используется первая голова, вторая не обязательна.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = backbone.fc.in_features

        # backbone возвращает фичи без финального FC
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # голова для распределения оценок 1..10
        self.fc_dist = nn.Linear(in_features, 10)

        # ординационная голова (можно игнорировать на инференсе)
        self.fc_ord = nn.Linear(in_features, 9)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits_dist = self.fc_dist(features)
        logits_ord = self.fc_ord(features)
        return logits_dist, logits_ord


# ===== Трансформация, как в get_eval_transform из finetune.py =====

eval_transform = T.Compose(
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


# ===== Загрузка модели (ленивая) =====

_model: ResNet50AvaMultiHead | None = None


def get_model() -> ResNet50AvaMultiHead:
    """
    Лениво создаёт и загружает модель ResNet50AvaMultiHead из чекпоинта.
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH}")

        model = ResNet50AvaMultiHead(pretrained=False).to(DEVICE)

        # В finetune.py сохранялся state_dict:
        # torch.save(best_state_dict, new_model_path)
        state = torch.load(MODEL_PATH, map_location=DEVICE)

        # Если вдруг чекпоинт в формате {"state_dict": ...}, можно раскомментировать:
        # if isinstance(state, dict) and "state_dict" in state:
        #     state = state["state_dict"]

        model.load_state_dict(state)
        model.eval()
        _model = model

    return _model


# ===== Инференс для одного изображения =====

@torch.no_grad()
def predict(image: Image.Image) -> Tuple[float, Dict[str, float]]:
    """
    На вход: PIL.Image.
    На выход:
      - нормализованный эстетический балл (0–10),
      - словарь { "1": p1, ..., "10": p10 } для отображения распределения.
    """
    if image is None:
        return 0.0, {}

    model = get_model()

    img = image.convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)

    logits_dist, logits_ord = model(tensor)

    probs = F.softmax(logits_dist, dim=1)[0]  # [10]

    ks = torch.arange(1, 11, dtype=torch.float32, device=probs.device)
    raw_mean_score = float((probs * ks).sum().item())  # mean оценки 1..10 по AVA

    human_score = normalize_to_human_scale(raw_mean_score)  # 0..10

    dist = {str(k): float(p) for k, p in zip(range(1, 11), probs.tolist())}

    return human_score, dist


# ===== Интерфейс Gradio =====

def build_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
# AVA Aesthetic Score (multi-head ResNet-50)

Устройство: **{DEVICE}**  
Модель: **{MODEL_PATH.name}**

Шкала в интерфейсе нормализована в диапазон **0–10**  
(внутри модель всё ещё работает в исходной шкале AVA 1–10).
            """
        )

        with gr.Row():
            input_image = gr.Image(
                type="pil",
                label="Изображение",
            )
            with gr.Column():
                score_output = gr.Number(
                    label="Нормализованный эстетический балл (0–10)"
                )
                dist_output = gr.Label(
                    label="Распределение по баллам (сырые вероятности модели, 1–10)",
                    num_top_classes=10,
                )

        run_btn = gr.Button("Оценить")

        run_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=[score_output, dist_output],
        )

        # Автоинференс при смене изображения
        input_image.change(
            fn=predict,
            inputs=input_image,
            outputs=[score_output, dist_output],
        )

    return demo


if __name__ == "__main__":
    iface = build_interface()
    iface.launch(share=False)
