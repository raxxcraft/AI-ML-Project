from pathlib import Path
import shutil
import kagglehub

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def copy_images(src: Path, dst: Path, prefix: str) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            out = dst / f"{prefix}_{n:06d}{p.suffix.lower()}"
            shutil.copy2(p, out)
            n += 1
    return n


def find_class_dirs(root: Path, class_names: set[str]) -> list[Path]:
    hits = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in class_names:
            hits.append(p)
    return hits


def main() -> None:
    raw_dir = Path(kagglehub.dataset_download("hazemfahmy/openned-closed-eyes")).resolve()
    print("Downloaded:", raw_dir)

    project_dir = Path(__file__).resolve().parent
    final_root = project_dir / "final_dataset"
    closed_out = final_root / "Closed"
    open_out = final_root / "Open"

    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)

    closed_count = 0
    open_count = 0

    closed_dirs = find_class_dirs(raw_dir, {"closed", "close"})
    open_dirs = find_class_dirs(raw_dir, {"open", "opened"})

    for p in closed_dirs:
        closed_count += copy_images(p, closed_out, "closed")
    for p in open_dirs:
        open_count += copy_images(p, open_out, "open")

    print("Prepared:", final_root)
    print("Closed images:", closed_count)
    print("Open images:", open_count)

    if closed_count == 0 or open_count == 0:
        print("Could not auto-find class folders. Check dataset folder names and map manually.")


if __name__ == "__main__":
    main()
