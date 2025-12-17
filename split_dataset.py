# split_dataset.py
import os
import random
import shutil
from pathlib import Path

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

SRC = Path("uniform_class_dataset")
DST = Path("dataset")

def split_class(src_class_dir: Path, dst_root: Path):
    files = [p for p in src_class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    clsname = src_class_dir.name
    for sub, filelist in [("train", train_files), ("val", val_files), ("test", test_files)]:
        out_dir = dst_root / sub / clsname
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in filelist:
            shutil.copy(f, out_dir / f.name)

def main():
    random.seed(RANDOM_SEED)
    if not SRC.exists():
        print("Source dataset folder not found:", SRC)
        return
    if DST.exists():
        print("Removing existing dataset/ folder (be careful)...")
        # optionally remove old or comment next two lines if you want to keep
        shutil.rmtree(DST)
    class_dirs = [d for d in SRC.iterdir() if d.is_dir()]
    for c in class_dirs:
        split_class(c, DST)
        print(f"Split class {c.name}")
    print("Done. Train/val/test in 'dataset/'")

if __name__ == "__main__":
    main()
