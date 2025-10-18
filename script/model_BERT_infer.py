# =============== model_BERT_infer.py ===============
import os, glob, json, math, numpy as np, pandas as pd, torch
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset
from utils import load_config   # 统一配置入口

#  1. 读 config 
cfg = load_config()

# 2. 解析路径 
CKPT_ROOT = cfg["bert_checkpoint"]          # ← 新增唯一键
TXT_IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    cfg["txt_test_cleaned_dir"]
)
OUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    cfg["csv_prediction_dir"],
    "model_BERT_prediction.csv"
)

MAX_LEN = cfg.get("bert_max_len", 512)
STRIDE  = cfg.get("bert_stride", 128)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  3. 工具函数 
def slide_window(text, max_len=MAX_LEN, stride=STRIDE):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    windows = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_len-2]
        if len(chunk) < 64:
            continue
        windows.append([tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id])
    return windows

#  4. 加载 tokenizer & model 
BASE = os.path.dirname(os.path.dirname(__file__))          # 项目根目录
CKPT_ABS = os.path.abspath(os.path.join(BASE, CKPT_ROOT))  # 绝对路径

tokenizer = BertTokenizerFast.from_pretrained(CKPT_ABS, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(
    CKPT_ABS, local_files_only=True
).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

#  9. 落盘 
def main():
    global tokenizer, model   # 让子进程能拿到对象（其实已序列化，可省）

    # 5. 组装待推理样本 
    txt_files = sorted(glob.glob(os.path.join(TXT_IN_DIR, "*.txt")))
    if not txt_files:
        raise RuntimeError(f'目录 {TXT_IN_DIR} 下未找到任何 txt 文件！')

    pool = []
    for f in txt_files:
        book_id, vol_id = os.path.basename(f)[:-4].split('_')
        book_id, vol_id = int(book_id), int(vol_id)
        text = open(f, encoding='utf-8').read()
        for w in slide_window(text):
            pool.append({'input_ids': w,
                         'book_id': book_id,
                         'vol_id': vol_id})

    # 6. 构造 Dataset 
    ds = Dataset.from_list(pool)
    # Windows 下必须 num_proc=0 或加 main 保护；这里直接单进程最快
    ds = ds.map(lambda x: {'input_ids': x['input_ids'][:MAX_LEN]}, num_proc=0)
    ds.set_format(type='torch', columns=['input_ids'])

    # 7. 推理 
    args = TrainingArguments(
        output_dir=CKPT_ABS,
        per_device_eval_batch_size=cfg.get("bert_batch_size", 8),
        fp16=True,
        dataloader_drop_last=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    logits = trainer.predict(ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    # 8. 窗口 → 卷级合并 
    infer_df = pd.DataFrame({
        'book_id': [s['book_id'] for s in pool],
        'vol_id':  [s['vol_id']  for s in pool],
        'prob':    probs
    })
    vol_df = (infer_df
              .groupby(['book_id', 'vol_id'], as_index=False)
              .agg(pred_prob=('prob', 'mean'))
              .assign(
                  filename=lambda x: x['book_id'].astype(str) + '_' + x['vol_id'].astype(str)
              )
              [['filename', 'book_id', 'vol_id', 'pred_prob']])

    #  9. 落盘 
    vol_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'BERT 推理完成，卷级概率已保存至 {OUT_CSV}')


if __name__ == '__main__':
    main()