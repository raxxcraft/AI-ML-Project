import argparse
import shutil
from pathlib import Path


VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def copy_images(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VALID_EXT:
            continue
        out = dst / f"{dst.name.lower()}_{count:06d}{p.suffix.lower()}"
        shutil.copy2(p, out)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final dataset with separate Closed/Open folders")
    parser.add_argument("--closed-src", type=str, required=True, help="Source folder containing closed-eye images")
    parser.add_argument("--open-src", type=str, required=True, help="Source folder containing open-eye images")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="final_dataset",
        help="Output dataset root with Closed/ and Open/",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_root = (base / args.out_dir).resolve()
    closed_out = out_root / "Closed"
    open_out = out_root / "Open"

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    closed_count = copy_images(Path(args.closed_src).resolve(), closed_out)
    open_count = copy_images(Path(args.open_src).resolve(), open_out)

    print(f"Output: {out_root}")
    print(f"Closed images: {closed_count}")
    print(f"Open images: {open_count}")
    if closed_count == 0 or open_count == 0:
        raise RuntimeError("Dataset incomplete: both Closed and Open must contain images.")


if __name__ == "__main__":
    main()
