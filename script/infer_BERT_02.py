# ===============  infer_BERT_02.py ===============
import json, os, glob, math, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset
import torch
from collections import Counter

# ========== 路径配置（只改这里） ==========
CKPT_ROOT     = r'train\models\checkpoint-15032'          # 已有 checkpoint
TXT_VAL_DIR   = r'train\assets\txt_cleaned'
LABEL_VAL_FILE= r'train\label\label.csv'
OUT_PROB_CSV  = r'train\csv\result02\val_window_prob_large.csv'   # 每窗口
OUT_PROB_VOL_CSV = r'train\csv\result02\val_vol_prob_large.csv'   # 每卷
OUT_METR_CSV  = r'train\csv\result02\val_metrics_large.csv'       # 指标
model_dir     = CKPT_ROOT

MAX_LEN = 512
STRIDE  = 128
os.makedirs(os.path.dirname(OUT_PROB_CSV), exist_ok=True)

# =================  函数定义（与训练脚本一致）  =================
def chunk_by_volume(book_id, txt_dir=TXT_VAL_DIR, max_char=150_000):
    vols = sorted(glob.glob(os.path.join(txt_dir, f"{book_id}_*.txt")))
    chunks = []
    for v in vols:
        vol_id = int(os.path.basename(v)[:-4].split('_')[1])
        txt = open(v, encoding='utf-8').read()
        for i in range(0, len(txt), max_char):
            chunks.append((vol_id, txt[i:i+max_char]))
    return chunks

def slide_window(text, max_len=MAX_LEN, stride=STRIDE):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    windows = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_len-2]
        if len(chunk) < 64:
            continue
        windows.append([tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id])
    return windows

def build_dataset(texts, labels, book_ids, vol_ids):
    all_win, all_lbl, all_bid, all_vid = [], [], [], []
    for text, lbl, bid, vid in zip(texts, labels, book_ids, vol_ids):
        for w in slide_window(text):
            all_win.append(w)
            all_lbl.append(int(lbl))
            all_bid.append(bid)
            all_vid.append(vid)
    ds = Dataset.from_dict({
        'input_ids': all_win,
        'labels': all_lbl,
        'book_id': all_bid,
        'vol_id': all_vid
    })
    ds = ds.map(lambda x: {'input_ids': x['input_ids'][:MAX_LEN]}, num_proc=1)
    ds.set_format(type='torch', columns=['input_ids', 'labels', 'book_id', 'vol_id'])
    return ds

# =================  主入口  =================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 读取验证标签（dropna 直接扔掉空标签行）
    val_labels_df = pd.read_csv(LABEL_VAL_FILE).dropna(subset=['label'])
    val_book2label = dict(zip(val_labels_df.book_id.astype(int), val_labels_df.label))

    # 2. 组装验证池（保留 book_id & vol_id）
    val_pool = []
    for bid in val_book2label.keys():
        # 如果 txt 文件不存在直接跳过
        if not glob.glob(os.path.join(TXT_VAL_DIR, f'{bid}_*.txt')):
            continue
        # 如果标签是 NaN 也跳过（double check）
        lbl = val_book2label[bid]
        if pd.isna(int(lbl)):          # NaN 直接跳过
            continue
        for vol_id, chk in chunk_by_volume(bid, TXT_VAL_DIR):
            val_pool.append((chk, lbl, bid, vol_id))

    # 3. 如果过滤后全空，及时提醒
    if not val_pool:
        raise RuntimeError('没有有效样本，请检查 label.csv 与 txt_cleaned 是否匹配！')

    val_txt, val_lbl, val_bid, val_vid = zip(*val_pool)

    # 3. 分词 & 数据集
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    val_ds = build_dataset(val_txt, val_lbl, val_bid, val_vid)

    # 4. 加载模型
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

    # 5. 构造 Trainer
    args = TrainingArguments(
        output_dir=CKPT_ROOT,
        per_device_eval_batch_size=8,
        fp16=True,
        dataloader_drop_last=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # 6. 预测
    pred_out = trainer.predict(val_ds)
    logits = pred_out.predictions
    probs_all = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    labels_all = pred_out.label_ids

    print('推理样本数:', len(val_ds))
    print('推理窗口数:', len(probs_all))

    # 7. 窗口级落盘
    infer_df = pd.DataFrame({
        'book_id': val_ds['book_id'],
        'vol_id': val_ds['vol_id'],
        'prob': probs_all,
        'label': labels_all
    })
    infer_df.to_csv(OUT_PROB_CSV, index=False, encoding='utf-8')
    print(f"每窗口概率已保存：{OUT_PROB_CSV}")

    # 8. 卷级合并（一卷一条）—— 按指定列名 & 拼 filename
    vol_df = (
        infer_df
        .groupby(['book_id', 'vol_id'], as_index=False)
        .agg(
            pred_prob=('prob', 'mean'),   # 对应脚本里的加权列名
            label=('label', 'first')
        )
        .assign(
            filename=lambda x: x['book_id'].astype(str) + '_' + x['vol_id'].astype(str).str.zfill(4)
        )
        # 调整列顺序
        [['filename', 'book_id', 'vol_id', 'pred_prob', 'label']]
    )
    vol_df.to_csv(OUT_PROB_VOL_CSV, index=False, encoding='utf-8')
    print(f"每卷合并结果已保存：{OUT_PROB_VOL_CSV}")
    #如果窗口级行数约等于卷级行数，可能合并失败，请使用check_vol_agg.py手动合并为卷级概率

    # 9. 指标
    acc_win = accuracy_score(labels_all, (probs_all > 0.5).astype(int))
    loss_win = log_loss(labels_all, probs_all)
    acc_vol = accuracy_score(vol_df['label'], (vol_df['prob'] > 0.5).astype(int))
    loss_vol = log_loss(vol_df['label'], vol_df['prob'])
    metrics_df = pd.DataFrame({
        'level': ['window', 'volume'],
        'accuracy': [acc_win, acc_vol],
        'log_loss': [loss_win, loss_vol]
    })
    metrics_df.to_csv(OUT_METR_CSV, index=False, encoding='utf-8')
    print(f"指标已保存：{OUT_METR_CSV}")
    print(f"ACC = {acc_win:.4f} , logloss={loss_win:.4f}")
    df = pd.read_csv(OUT_PROB_CSV)
    print('AUC =', roc_auc_score(df.label, df.prob))

    #这里acc仅使用了默认阈值0.5，auc作为更有参考价值