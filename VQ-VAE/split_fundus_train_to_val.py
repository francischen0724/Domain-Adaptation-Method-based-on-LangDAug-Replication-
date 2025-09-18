import argparse, os, random, shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def list_pairs(domain_dir: Path):
    img_dir = domain_dir / "image"
    mask_dir = domain_dir / "mask"
    rois_dir = domain_dir / "ROIs"  # 可选

    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expect image/ and mask/ under {domain_dir}")

    # 以 image 下的文件为基准，按不带扩展名的 basename 配对
    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    pairs = []
    for img in imgs:
        base = img.stem  # 不含扩展名
        # 在 mask 中寻找相同 basename 的文件（任意图片后缀）
        mask_candidates = list(mask_dir.glob(base + ".*"))
        if not mask_candidates:
            print(f"[WARN] mask missing for {img}")
            continue
        mask = mask_candidates[0]
        roi = None
        if rois_dir.is_dir():
            roi_candidates = list(rois_dir.glob(base + ".*"))
            if roi_candidates:
                roi = roi_candidates[0]
        pairs.append((img, mask, roi))
    return pairs, rois_dir.is_dir()

def ensure_dirs(root: Path, split: str, domain: str, with_rois: bool):
    base = root / split / domain
    (base / "image").mkdir(parents=True, exist_ok=True)
    (base / "mask").mkdir(parents=True, exist_ok=True)
    if with_rois:
        (base / "ROIs").mkdir(parents=True, exist_ok=True)
    return base

def move_or_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

def process_domain(root: Path, domain: str, val_ratio: float, seed: int, copy: bool):
    train_domain = root / "train" / domain
    val_domain_base = root / "val" / domain

    pairs, has_rois = list_pairs(train_domain)
    if not pairs:
        print(f"[SKIP] No pairs in {train_domain}")
        return

    random.Random(seed).shuffle(pairs)
    k = max(1, int(len(pairs) * val_ratio))
    val_set = set(pairs[:k])

    # 预创建 val 目录
    ensure_dirs(root, "val", domain, has_rois)

    moved = 0
    for img, mask, roi in val_set:
        # 目标路径
        img_dst = root / "val" / domain / "image" / img.name
        mask_dst = root / "val" / domain / "mask"  / mask.name
        move_or_copy(img,  img_dst, copy)
        move_or_copy(mask, mask_dst, copy)
        if roi is not None:
            roi_dst = root / "val" / domain / "ROIs" / roi.name
            move_or_copy(roi, roi_dst, copy)
        moved += 1

    print(f"[OK] {domain}: moved_to_val={moved} / total_train_pairs={len(pairs)} (ratio={val_ratio})  copy={copy}")

def main():
    ap = argparse.ArgumentParser(description="Split fundus train into val per domain (by filenames).")
    ap.add_argument("--fundus_root", required=True, help="path to fundus root (contains train/, test/)")
    ap.add_argument("--domains", nargs="*", default=None,
                    help="domains to split (default: auto-detect from train/)")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="portion of train moved to val per domain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="copy instead of move (default move)")
    args = ap.parse_args()

    root = Path(args.fundus_root).resolve()
    if not (root / "train").is_dir():
        raise FileNotFoundError(f"{root}/train not found")

    if args.domains is None:
        args.domains = sorted([d.name for d in (root / "train").iterdir() if d.is_dir()])

    (root / "val").mkdir(exist_ok=True)

    print(f"Fundus root: {root}")
    print(f"Domains: {args.domains}")
    for dom in args.domains:
        process_domain(root, dom, args.val_ratio, args.seed, args.copy)

if __name__ == "__main__":
    main()

## run
# python split_fundus_train_to_val.py \
#   --fundus_root ./datasets/fundus \
#   --val_ratio 0.2 \
#   --seed 42
