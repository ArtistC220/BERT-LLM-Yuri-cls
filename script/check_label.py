import pandas as pd, numpy as np, os, glob

TXT_VAL_DIR   = r'train\assets\txt_cleaned'
LABEL_VAL_FILE= r'train\label\label.csv'

val_labels_df = pd.read_csv(LABEL_VAL_FILE)
print('label.csv 缺失值统计：')
print(val_labels_df.isna().sum())

# 所有在 txt 目录里出现过的 book_id
txt_books = set()
for f in glob.glob(os.path.join(TXT_VAL_DIR, '*.txt')):
    # 文件名形如  12345_1.txt
    bid = int(os.path.basename(f).split('_')[0])
    txt_books.add(bid)

# 所有在 label.csv 里出现过的 book_id
label_books = set(val_labels_df['book_id'].astype(int))

print('在 txt 里有，但 label.csv 里缺失的 book_id：')
miss = txt_books - label_books
print(miss)

print('在 label.csv 里有，但 txt 里缺失的 book_id：')
extra = label_books - txt_books
print(extra)