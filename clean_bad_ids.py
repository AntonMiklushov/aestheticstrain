from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
AVA_ROOT  = DATA_ROOT / "AVA"

# ВАЖНО: здесь именно aesthetics_image_lists, как в finetune.py
LISTS_DIR = AVA_ROOT / "aesthetics_image_lists"

BAD_IDS = {"230701", "104855", "848725", "953980", "277832"}

def normalize_id(line: str) -> str:
    s = line.strip()
    s = s.split("/")[-1].split("\\")[-1]
    if s.lower().endswith(".jpg"):
        s = s[:-4]
    return s

def clean_list(filename: str):
    path = LISTS_DIR / filename
    lines = path.read_text(encoding="utf-8").splitlines()

    kept = []
    removed = []
    for line in lines:
        if not line.strip():
            continue
        img_id = normalize_id(line)
        if img_id in BAD_IDS:
            removed.append(line)
        else:
            kept.append(line)

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text("\n".join(lines), encoding="utf-8")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(f"{filename}: оставлено {len(kept)}, удалено {len(removed)}")
    if removed:
        print("Удалено строки:")
        for r in removed:
            print("  ", r)

if __name__ == "__main__":
    for name in ["generic_ls_train.jpgl", "generic_ss_train.jpgl", "generic_test.jpgl"]:
        clean_list(name)
