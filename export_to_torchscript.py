# export_to_torchscript.py
#
# Преобразует уже обученный чекпоинт
#   models/resnet50_ava_regression_finetune_001.pt
# в TorchScript-модель
#   models/resnet50_ava_regression_finetune_001_mobile.pt
#
# Никакого обучения тут нет: только загрузка весов и экспорт.

from pathlib import Path

import torch

# Берём определение модели и константы из лёгкого инференс-скрипта
from run_inference_gradio import ResNet50AvaMultiHead, IMG_SIZE, MODEL_PATH


def main():
    device = torch.device("cpu")

    # 1. Создаём такую же модель, как при инференсе
    model = ResNet50AvaMultiHead(pretrained=False)
    model.to(device)

    # 2. Загружаем state_dict из существующего файла
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден чекпоинт: {MODEL_PATH}")

    print(f"Загружаю веса из: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)

    # На всякий случай: если вдруг когда-нибудь будет формат {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    # 3. Примерный входной тензор: [1, 3, IMG_SIZExIMG_SIZE]
    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    # 4. TorchScript через trace (forward у Вас без if'ов, это безопасно)
    print("Делаю torch.jit.trace(...)")
    scripted = torch.jit.trace(model, example_input)
    scripted = scripted.cpu()

    # 5. Имя выходного файла
    ts_path: Path = MODEL_PATH.with_name(MODEL_PATH.stem + "_mobile.pt")

    scripted.save(str(ts_path))
    print(f"Готово. TorchScript-модель сохранена в: {ts_path}")


if __name__ == "__main__":
    main()
