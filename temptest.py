import torch
from pathlib import Path

from run_inference_gradio import IMG_SIZE, eval_transform
from PIL import Image

ts_path = Path("models/resnet50_ava_regression_finetune_001_mobile.pt")
ts_model = torch.jit.load(ts_path, map_location="cpu")
ts_model.eval()

img = Image.open("data/AVA/images/198.jpg").convert("RGB")
tensor = eval_transform(img).unsqueeze(0)  # [1,3,448,448]

with torch.no_grad():
    logits_dist, logits_ord = ts_model(tensor)
    print(logits_dist.shape)  # ожидаете [1, 10]
