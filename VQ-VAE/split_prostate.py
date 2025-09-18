# split_prostate_train_val_test.py
import os, re, random, shutil
from pathlib import Path
random.seed(0)

root = Path("./datasets/prostate")
domains = ["BIDMC","BMC","HK","I2CVB","RUNMC","UCL"]

SEG_RE = re.compile(r"(_seg(mentation)?)$", re.IGNORECASE)
def strip_ext(p): 
    s = p.name
    return s[:-7] if s.lower().endswith(".nii.gz") else s[:-4] if s.lower().endswith(".nii") else os.path.splitext(s)[0]

def collect_pairs(dom_dir: Path):
    imgs, msks = {}, {}
    files = [f for f in dom_dir.iterdir() if f.suffix.lower() in [".nii",".gz"]]
    for f in files:
        stem = strip_ext(f)
        if SEG_RE.search(stem):
            base = SEG_RE.sub("", stem)
            msks[base] = f
        else:
            imgs[stem] = f
    pairs = []
    for base, img in imgs.items():
        m = msks.get(base)
        if m: pairs.append((img, m, base))
    return sorted(pairs, key=lambda x:x[2])

# 配置：每个域内部 train:val:test = 70%:10%:20%
train_ratio, val_ratio = 0.6, 0.2

for d in domains:
    dom_dir = root / d
    pairs = collect_pairs(dom_dir)
    pids  = [b for _,_,b in pairs]
    uniq  = sorted(set(pids))
    n     = len(uniq)
    n_train = int(n*train_ratio)
    n_val   = int(n*val_ratio)

    train_ids = set(uniq[:n_train])
    val_ids   = set(uniq[n_train:n_train+n_val])
    test_ids  = set(uniq[n_train+n_val:])

    def move_pair(split_name, img, msk, base):
        out_img = root / split_name / d / "image"
        out_msk = root / split_name / d / "mask"
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out_img / img.name)
        shutil.copy2(msk, out_msk / msk.name)

    for img, msk, base in pairs:
        if base in train_ids: move_pair("train", img, msk, base)
        elif base in val_ids: move_pair("val",   img, msk, base)
        else:                 move_pair("test",  img, msk, base)

    print(f"{d}: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
