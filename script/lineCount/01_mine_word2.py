# 04_diff_dialog_words.py
import os, pandas as pd, re, jieba
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = r'C:\Users\19716\OneDrive\Desktop\杂\py'
CSV_PATH = os.path.join(BASE_DIR, r'train\csv\prediction_version06.csv')
TXT_DIR  = os.path.join(BASE_DIR, r'train\txt')
OUT_PATH = os.path.join(BASE_DIR, r'train\script\lineCount\diff_dialog_words.txt')

# ---------- 1. 读取高/低置信卷 ----------
label = pd.read_csv(CSV_PATH)
high = label.nlargest(50, 'pred_prob')
low  = label.nsmallest(50, 'pred_prob')

# ---------- 2. 只提取「对话句」 ----------
DIALOG_RE = re.compile(r'([「『].*?[」』])')          # 引号内
SPEAK_RE  = re.compile(r'(.+?)(?:说|道|问|答|喊|叫|嘀咕|低声|怒道|冷声道)(?=：|「|』)')  # 说/道/问/答 后接冒号或引号

STOP_WORDS = set('的 了 我 你 她 是 在 有 和 就 都 不 也 而 却 但 如果 因为 所以 可是 不过 吧 啊 呢 吗 着 地 得 了 的'.split())

def extract_dialog(text):
    # 1. 引号内
    quotes = DIALOG_RE.findall(text)
    # 2. 说/道/问/答 后接冒号或引号
    speaks = SPEAK_RE.findall(text)
    dialog_lines = quotes + speaks
    # 3. 分词 + 停 + 单字
    words = []
    for sent in dialog_lines:
        words += [w for w in jieba.lcut(sent) if len(w) > 1 and w not in STOP_WORDS]
    return ' '.join(words)

def read_dialog(fname):
    fp = os.path.join(TXT_DIR, fname)
    if not os.path.exists(fp):
        return ''
    with open(fp, encoding='utf-8') as f:
        return extract_dialog(f.read())

# ---------- 3. 拼对话语料 ----------
docs_A = [read_dialog(r) for r in high['filename']]
docs_B = [read_dialog(r) for r in low['filename']]
all_docs = docs_A + docs_B

# ---------- 4. TF-IDF（1-2 元组，min_df=2） ----------
vec = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),
    min_df=2,
    lowercase=False,
    tokenizer=lambda s: s.split()   # 已分词好
)

X = vec.fit_transform(all_docs)
id2word = {i: w for i, w in enumerate(vec.get_feature_names_out())}

# ---------- 5. 取 A（前 50 篇）平均权重 ----------
a_weights = X[:50].mean(axis=0).A1
top_idx = a_weights.argsort()[-100:][::-1]

# ---------- 6. 输出 ----------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    for i in top_idx:
        w = id2word[i]
        f.write(f'{w}\t{a_weights[i]:.4f}\n')

print('对话差异独占词 100 条已写入', OUT_PATH)