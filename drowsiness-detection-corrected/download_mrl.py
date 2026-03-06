from __future__ import annotations

import argparse
import shutil
from pathlib import Path

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def all_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT]


def infer_label_index(images: list[Path]) -> int | None:
    # MRL filenames often contain multiple underscore-separated attributes with a 0/1 eye-state field.
    # Auto-select the token position that best separates samples into both classes.
    if not images:
        return None

    max_tokens = max(len(p.stem.split("_")) for p in images)
    best_idx = None
    best_score = -1

    for idx in range(max_tokens):
        zeros = 0
        ones = 0
        covered = 0
        for p in images:
            parts = p.stem.split("_")
            if idx >= len(parts):
                continue
            t = parts[idx].strip()
            if t == "0":
                zeros += 1
                covered += 1
            elif t == "1":
                ones += 1
                covered += 1

        if covered == 0:
            continue
        # Favor positions that cover many files and include both classes.
        score = covered + min(zeros, ones)
        if zeros > 0 and ones > 0 and score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def label_from_path(p: Path, idx: int | None) -> str | None:
    # Folder-name fallback first.
    low_parts = [x.lower() for x in p.parts]
    if any("close" in x for x in low_parts):
        return "Closed"
    if any("open" in x for x in low_parts):
        return "Open"

    if idx is None:
        return None
    parts = p.stem.split("_")
    if idx >= len(parts):
        return None
    t = parts[idx].strip()
    if t == "0":
        return "Closed"
    if t == "1":
        return "Open"
    return None


def prepare_dataset(source: Path, out_root: Path) -> tuple[int, int]:
    imgs = all_images(source)
    if not imgs:
        raise RuntimeError(f"No images found in: {source}")

    idx = infer_label_index(imgs)
    print(f"Detected label token index: {idx}")

    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "Closed").mkdir(parents=True, exist_ok=True)
    (out_root / "Open").mkdir(parents=True, exist_ok=True)

    c = 0
    o = 0
    for p in imgs:
        lab = label_from_path(p, idx)
        if lab == "Closed":
            dst = out_root / "Closed" / f"closed_{c:06d}{p.suffix.lower()}"
            shutil.copy2(p, dst)
            c += 1
        elif lab == "Open":
            dst = out_root / "Open" / f"open_{o:06d}{p.suffix.lower()}"
            shutil.copy2(p, dst)
            o += 1

    return c, o


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MRL dataset from Kaggle and prepare Closed/Open folders")
    parser.add_argument("--dataset", type=str, default="prasadvpatil/mrl-dataset")
    parser.add_argument("--source", type=str, default="", help="Optional existing source path (skip download)")
    parser.add_argument("--out-dir", type=str, default="final_mrl_dataset")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    if args.source:
        source = Path(args.source).resolve()
    else:
        import kagglehub

        source = Path(kagglehub.dataset_download(args.dataset)).resolve()

    print(f"Source dataset path: {source}")

    out_root = (base / args.out_dir).resolve()
    c, o = prepare_dataset(source, out_root)
    print(f"Prepared dataset: {out_root}")
    print(f"Closed images: {c}")
    print(f"Open images: {o}")

    if c == 0 or o == 0:
        raise RuntimeError("Failed to build both classes. Please inspect source naming and adjust mapping.")


if __name__ == "__main__":
    main()
