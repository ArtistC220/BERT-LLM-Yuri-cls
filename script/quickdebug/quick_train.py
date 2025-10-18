# ================= 1. 全局库 =================
import json, os, glob, random, numpy as np, pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset
from collections import Counter
import torch
from sklearn.metrics import roc_auc_score

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ========== 路径配置（只改这里） ==========
TXT_TRAIN_DIR = r'train\assets\txt_train_cleaned1'
LABEL_FILE    = r'train\label\label.csv'
model_dir     = r'train\models\chinese-roberta-wwm-ext'

# ========== 函数定义 ==========
def slide_window(text, max_len=512, stride=128):
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    # 把 int 列表转空格分隔字符串
    text_str = ' '.join(map(str, text))
    tokens = tokenizer.encode(text_str, add_special_tokens=False)
    windows = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_len-2]
        if len(chunk) < 64: continue
        windows.append([101] + chunk + [102])
    return windows

def build_dataset(texts, labels, vol_ids):
    all_win, all_lbl, all_vid = [], [], []
    for text, lbl, vid in zip(texts, labels, vol_ids):
        for w in slide_window(text):
            all_win.append(w); all_lbl.append(int(lbl)); all_vid.append(vid)
    ds = Dataset.from_dict({'input_ids': all_win, 'labels': all_lbl, 'vol_id': all_vid})
    ds = ds.map(lambda x: {'input_ids': x['input_ids'][:512]}, num_proc=1)
    ds.set_format(type='torch', columns=['input_ids', 'labels', 'vol_id'])
    return ds

# ========== 主入口（Windows 必须） ==========
if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读标签 & 快速采样 100 条
    labels_df = pd.read_csv(LABEL_FILE)
    book2label = dict(zip(labels_df.book_id.astype(int), labels_df.label))
    pool = []
    for bid in book2label.keys():
        for v in sorted(glob.glob(os.path.join(TXT_TRAIN_DIR, f"{bid}_*.txt"))):
            txt = open(v, encoding='utf-8').read()[:150_000]
            for w in slide_window(txt):
                pool.append((w, book2label[bid], bid))
    random.seed(42); pool = random.sample(pool, 100)   # 只抽 100 条
    train_txt, train_lbl, train_vid = zip(*pool)

    # 权重采样（每卷等权）
    from collections import Counter
    vol_counter = Counter(train_vid)
    weights = [1.0 / vol_counter[vid] for vid in train_vid]

    # 分词 & 数据集
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    train_ds = build_dataset(train_txt, train_lbl, train_vid)
    train_ds = train_ds.remove_columns(['vol_id'])   # ← 去掉 vol_id，不传模型

    # 快速训练参数（100 步，2-3 min）
    args = TrainingArguments(
        output_dir=r'train\models\ckpt_quick',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        max_steps=100,                     # 只跑 100 步
        learning_rate=1e-5,
        weight_decay=0.001,
        eval_strategy='steps',
        eval_steps=20,
        logging_steps=10,
        fp16=True,
        warmup_ratio=0.2,
    )

    def compute_metrics(pred):
        logits, labels = pred
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        print('>>> 前 20 个 prob:', probs[:20])
        print('>>> 最小 prob:', probs.min(), ' 最大 prob:', probs.max())
        return {'auc': roc_auc_score(labels, probs)}

    # 自定义 Trainer（带卷 ID 打印）
    class QuickTrainer(Trainer):
        def get_train_dataloader(self):
            dl = DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=WeightedRandomSampler(weights, len(self.train_dataset), replacement=True),
                collate_fn=DataCollatorWithPadding(tokenizer),
                drop_last=True,
                pin_memory=True,
            )
            # 打印全部抽到的卷 ID（每轮）
            sampled_vids = []
            for idx in range(len(self.train_dataset)):
                sampled_vids.append(self.train_dataset[idx]['vol_id'])
            print('>>> 全部抽到的卷 ID:', Counter(sampled_vids))
            return dl

    # 加载模型（不保存）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    train_ds = build_dataset(train_txt, train_lbl, train_vid)

    trainer = QuickTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print('>>> CUDA 可用:', torch.cuda.is_available())
    print('>>> 使用设备:', trainer.args.device)
    print('>>> 开始快速训练（100 步，约 2-3 min）...')
    trainer.train()
    print('>>> 训练完成！')