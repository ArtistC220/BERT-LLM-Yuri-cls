
"""
这是用于手动挑选训练集的脚本

按“整本书”维度抽取测试集，防止数据泄漏

"""
import os, shutil, csv, math, sys
import random

# ========== 可手动改的 5 个参数 ==========
SRC_DIR        = r'train\txt_train'        # 原始 txt 目录
OUT_DIR        = r'train\txt_BERT_verify1'  # 测试集（整本书）
UNUSED_DIR     = r'train\txt_BERT_train1'   # 训练集（整本书）
MIN_VOLUMES    = 3                         # 少于该卷数的书整本跳过
TARGET_TOTAL   = 200                      # 目标测试集卷数
# ========================================

LABEL_FILE     = r'train\label\label.csv'
VOLUME_FILE    = r'train - 副本\csv\result\volume_result.csv'

# ----------- 工具函数 -----------
def read_label(path):
    label_map = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            bid = int(row['book_id'])
            lab = row['label'].strip()
            label_map[bid] = int(lab) if lab in {'0','1'} else 0
    return label_map

def read_volumes(path):
    vol_map = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            bid = int(row['aid'])
            vid = int(row['vid'])
            vol_map.setdefault(bid, []).append(vid)
    for bid in vol_map:
        vol_map[bid] = sorted(set(vol_map[bid]))
    return vol_map

def pick_whole_books(vol_map, label_map, min_v, target):
    """返回 (picked_books, unpicked_books) 两个 book_id 集合"""
    qualified = {bid: vids for bid, vids in vol_map.items() if len(vids) >= min_v}
    total_avail = sum(len(vids) for vids in qualified.values())
    need_books = max(1, int(math.floor(len(qualified) * target / total_avail)))

    # 分层：0 与 1 两组
    label0_books = [bid for bid in qualified if label_map.get(bid, 0) == 0]
    label1_books = [bid for bid in qualified if label_map.get(bid, 0) == 1]

    # 内部随机打乱
    random.shuffle(label0_books)
    random.shuffle(label1_books)

    # 尽量 1:1 抽取
    half = need_books // 2
    pick0 = label0_books[:half]
    pick1 = label1_books[:need_books - len(pick0)]   # 剩下名额给 1
    picked_books = set(pick0 + pick1)

    # 其余合格书 + 不合格书 全部归训练集
    unpicked_books = set(qualified.keys()) - picked_books
    for bid in vol_map:
        if bid not in qualified:
            unpicked_books.add(bid)

    return picked_books, unpicked_books

def copy_books(book_set, src_dir, dst_dir, tag=''):
    os.makedirs(dst_dir, exist_ok=True)
    cnt = 0
    for bid in book_set:
        for vid in vol_map[bid]:
            fname = f'{bid}_{vid}.txt'
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(src):
                fname2 = f'{bid}_{vid:04d}.txt'
                src2 = os.path.join(src_dir, fname2)
                if os.path.exists(src2):
                    src = src2
                else:
                    print(f'[WARN] 文件不存在，跳过: {src} 或 {src2}', file=sys.stderr)
                    continue
            shutil.copy2(src, dst)
            cnt += 1
    print(f'{tag} 复制完成，共 {cnt} 卷 -> {dst_dir}')

# ---------- 主流程 ----------
def main():
    if not os.path.exists(SRC_DIR):
        print(f'源目录不存在: {SRC_DIR}')
        return
    global vol_map
    label_map = read_label(LABEL_FILE)
    vol_map   = read_volumes(VOLUME_FILE)
    picked_books, unpicked_books = pick_whole_books(vol_map, label_map, MIN_VOLUMES, TARGET_TOTAL)
    total_picked_vols = sum(len(vol_map[bid]) for bid in picked_books)
    print(f'计划抽取 {len(picked_books)} 本书（共 {total_picked_vols} 卷）作为测试集')
    copy_books(picked_books,   SRC_DIR, OUT_DIR,    '测试集')
    copy_books(unpicked_books, SRC_DIR, UNUSED_DIR, '训练集')

if __name__ == '__main__':
    main()