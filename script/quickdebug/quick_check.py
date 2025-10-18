# quick_check.py
import json, os, glob, random, numpy as np, pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import Dataset
from sklearn.metrics import roc_auc_score
import torch

# ========== 同目录路径 ==========
TXT_TRAIN_DIR = r'train\assets\txt_train_cleaned1'
LABEL_FILE    = r'train\label\label.csv'
model_dir     = r'train\models\chinese-roberta-wwm-ext'

# ========== 同目录函数（最小化） ==========
def slide_window(text, max_len=512, stride=128):
    tokens = BertTokenizerFast.from_pretrained(model_dir).encode(text, add_special_tokens=False)
    windows = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_len-2]
        if len(chunk) < 64: continue
        windows.append([101] + chunk + [102])
    return windows

def quick_sample(n=100):
    labels_df = pd.read_csv(LABEL_FILE)
    book2label = dict(zip(labels_df.book_id.astype(int), labels_df.label))
    pool = []
    for bid in book2label.keys():
        for v in sorted(glob.glob(os.path.join(TXT_TRAIN_DIR, f"{bid}_*.txt"))):
            txt = open(v, encoding='utf-8').read()[:150_000]
            for w in slide_window(txt):
                pool.append((w, book2label[bid]))
    random.seed(42); return random.sample(pool, min(n, len(pool)))

# ========== 加载模型（不训练） ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = BertTokenizerFast.from_pretrained(model_dir)

# ========== 快速推理 100 条 ==========
sample_pool = quick_sample(100)
inputs = tokenizer(
    [' '.join(map(str, w)) for w, _ in sample_pool],  # ← 关键修复
    padding=True, truncation=True, max_length=512, return_tensors='pt'
).to(device)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    labels = [lbl for _, lbl in sample_pool]

# ========== 打印结果 ==========
print('>>> 前 20 个 prob:', probs[:20])
print('>>> 最小 prob:', probs.min(), ' 最大 prob:', probs.max())
print('>>> 快速 AUC (100 条):', roc_auc_score(labels, probs))