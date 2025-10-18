# 01_mine_words.py
import pandas as pd, glob, re, jieba, collections, os

# 1. 绝对路径（按你实际目录写）
BASE_DIR = r'C:\Users\19716\OneDrive\Desktop\杂\py'
CSV_PATH   = os.path.join(BASE_DIR, r'train\csv\prediction_version06.csv')
TXT_DIR    = os.path.join(BASE_DIR, r'train\txt')
OUT_PATH   = os.path.join(BASE_DIR, r'train\script\lineCount\candidate_words.txt')

# 2. 读取高独占卷（概率≥0.6）
label = pd.read_csv(CSV_PATH)
high  = label[label['pred_prob'] >= 0.6]

# 3. 拼大语料（去掉多余的 .txt）
texts = []
for _, row in high.iterrows():
    # 去掉文件名里的 .txt 后缀
    fname = row['filename'].replace('.txt', '')
    fp = os.path.join(TXT_DIR, f'{fname}.txt')
    if not os.path.exists(fp):
        print('跳过无文件:', fp)
        continue
    with open(fp, encoding='utf-8') as f:
        texts.append(f.read())
corpus = '\n'.join(texts)

# 4. 分词 + 停用
stop = set('的 了 我 你 她 是 在 有 和 就 都 不 也 而 却 但 如果 因为 所以 可是 不过'.split())
words = [w for w in jieba.lcut(corpus) if w not in stop and len(w) > 1]

# 5. 词频统计
freq = collections.Counter(words)
top200 = freq.most_common(200)

# 6. 输出
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    for w, c in top200:
        f.write(f'{w}\t{c}\n')