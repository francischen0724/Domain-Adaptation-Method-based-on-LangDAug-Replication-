import lmdb
import pickle
import os
from collections import defaultdict

def inspect_lmdb(lmdb_path, case_filter=None, max_print=50):
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"❌ LMDB 文件不存在: {lmdb_path}")

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    txn = env.begin()

    # 统计所有 key
    keys = []
    for k, _ in txn.cursor():
        keys.append(k.decode("utf-8"))

    print(f"✅ LMDB 打开成功: {lmdb_path}")
    print(f"🔢 总共存储了 {len(keys)} 个 keys")

    # 如果只想看某个 case，比如 BIDMC_Case00
    if case_filter:
        keys = [k for k in keys if k.startswith(case_filter)]
        print(f"📂 筛选 {case_filter}: {len(keys)} 个 key")

    # 排序（自然顺序，避免 _10 排在 _2 前）
    import re
    def natural_key(s):
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
    keys_sorted = sorted(keys, key=natural_key)

    # 打印前几个 key
    print(f"👀 前 {max_print} 个 key:")
    for k in keys_sorted[:max_print]:
        print("   ", k)

    # 打印后几个 key
    if len(keys_sorted) > max_print:
        print("...")
        print(f"👀 最后 {max_print} 个 key:")
        for k in keys_sorted[-max_print:]:
            print("   ", k)

    env.close()

if __name__ == "__main__":
    lmdb_path = "./datasets/prostate/data.lmdb"  # 修改成你的 LMDB 路径
    # 例如只看 BIDMC_Case00
    inspect_lmdb(lmdb_path, case_filter="BIDMC_Case00", max_print=20)
