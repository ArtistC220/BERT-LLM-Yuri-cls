# ================= 1. 全局库 =================
import json, os, glob, random, numpy as np, pandas as pd,math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback)
from datasets import Dataset
import torch, tqdm, time
from torch.utils.data import WeightedRandomSampler
from transformers import BertConfig

from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DataCollatorWithPadding
from collections import Counter

#第三版，加入卷级权重，出现过拟合 在train\models\ckpt4

#cpkt6 最佳auc 0.78
#cpkt4 多轮auc 0.74 
#cpkt8 多轮auc 0.728
#ckpt9 无采样 
#目前为大模型状态，单epoch 6h，尝试在04优化

class WeightedTrainer(Trainer):   #带加权采样的训练器
    """让 Trainer 使用 WeightedRandomSampler"""
    def __init__(self, train_sampler_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sampler_weights = train_sampler_weights

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=WeightedRandomSampler(
                self.train_sampler_weights,
                len(self.train_dataset),
                replacement=False
            ),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ========== 路径配置（只改这里） ==========
CONFIG_FILE   = r'train\config.json'
TXT_TRAIN_DIR = r'train\assets\txt_train_cleaned1'   # ← 训练
TXT_VAL_DIR   = r'train\assets\txt_val_cleaned1'     # ← 外部验证
LABEL_FILE    = r'train\label\label.csv'
LABEL_VAL_FILE= r'train\label\label.csv'        # ← 外部验证标签
CKPT_ROOT     = r'train\models\ckpt8'
RESULT_ROOT   = r'train\csv\result02'
model_dir     = r'train\models\chinese-roberta-wwm-ext'

os.makedirs(CKPT_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

MAX_CHAR      = 150_000          # 切块，滑窗，重叠的所有数据在这里改
MAX_LEN       = 512
STRIDE        = 128

#预统计窗口数的经验值
# 经验值：RoBERTa-wwm-ext 中文  1 汉字 → 0.75 token
CHAR2TOKEN_RATIO   = 0.75
# UTF-8 中文  1 汉字 → 3 bytes
BYTE2CHAR_RATIO    = 1 / 3
# ========== 函数定义 ==========
def prescan_windows(txt_dir = TXT_TRAIN_DIR):
    """
    只扫文件名+文件大小，返回 dict vol_id -> 该卷预计总窗口数
    适合中文小说，不打开文件，O(1) 额外 IO
    """
    vol2nwins = {}
    for fp in glob.glob(os.path.join(txt_dir, "*.txt")):
        fname   = os.path.basename(fp)           # 123_45.txt
        vol_id  = int(fname[:-4].split('_')[1])
        byte_sz = os.path.getsize(fp)            # 字节数

        # 1. 字节 → 字符
        nchar_total = byte_sz * BYTE2CHAR_RATIO
        # 2. 字符 → token
        ntok_total  = nchar_total * CHAR2TOKEN_RATIO
        # 3. 按 chunk 大小分段
        nchunk      = max(1, math.ceil(ntok_total / MAX_CHAR))
        nwins       = 0
        for _ in range(nchunk):
            chunk_tok = min(MAX_CHAR, ntok_total)
            # 4. 滑动窗口公式：ceil((L - W) / S) + 1，且过滤过短窗口
            if chunk_tok >= 64:                   # 你主脚本里 <64 会 continue
                wins = max(0, (chunk_tok - (MAX_LEN-2)) // STRIDE + 1)
                nwins += wins
            ntok_total -= MAX_CHAR
            if ntok_total <= 0:
                break
        vol2nwins[vol_id] = vol2nwins.get(vol_id, 0) + nwins
    return vol2nwins

def chunk_by_volume(book_id, txt_dir=TXT_TRAIN_DIR, max_char=MAX_CHAR ):
    vols = sorted(glob.glob(os.path.join(txt_dir, f"{book_id}_*.txt"))) #集合所有bookid对应的vol
    chunks = [] #集合所有bookid对应的vol
    for v in vols:
        vol_id = int(os.path.basename(v)[:-4].split('_')[1]) #每一卷切出vid
        txt = open(v, encoding='utf-8').read()
        for i in range(0, len(txt), max_char):
            chunks.append((vol_id, txt[i:i+max_char]))
    return chunks               #chunk是一本书输入 chunks是带（vol_id， 文本块）的元组

def slide_window(text, max_len=MAX_LEN, stride=STRIDE):
    tokens = tokenizer.encode(text, add_special_tokens=False) #将文本按token编码
    windows = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_len-2] #按token切chunk
        if len(chunk) < 64: continue
        windows.append([tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id])#chunk加入windows中
    return windows #返回对应文本切好的窗口

def build_dataset(texts, labels): #输入texts 和 标签（后面处理时会保证对应）
    all_win, all_lbl = [], []  #所有窗口 所有标签
    for text, lbl in zip(texts, labels):  #按文本块，标签遍历
        for w in slide_window(text): #文本切窗口，对每个窗口
            all_win.append(w); all_lbl.append(int(lbl))         #修改为整数类型
    ds = Dataset.from_dict({'input_ids': all_win, 'labels': all_lbl})  #组装dataset
    
    ds = ds.map(lambda x: {'input_ids': x['input_ids'][:MAX_LEN]}, num_proc=1) #截断窗口
    ds.set_format(type='torch', columns=['input_ids', 'labels'])          #声明文件格式
    return ds #返回dataset，可以直接跑训练

# ========== 主入口（Windows 必须） ==========
if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读配置 & 标签
    cfg = json.load(open(CONFIG_FILE, encoding='utf-8'))
    labels_df = pd.read_csv(LABEL_FILE)
    book2label = dict(zip(labels_df.book_id.astype(int), labels_df.label))


    vol_counter = Counter()
    train_pool  = []
    for bid in book2label.keys():
        for vol_id, chk in chunk_by_volume(bid, TXT_TRAIN_DIR):
            train_pool.append((chk, book2label[bid], bid, vol_id))
            vol_counter[vol_id] += 1          # 顺手计数
    vol_weight = {vid: 1.0 / c for vid, c in vol_counter.items()}
       
    
    val_labels_df = pd.read_csv(LABEL_VAL_FILE)
    val_book2label = dict(zip(val_labels_df.book_id.astype(int), val_labels_df.label))
    val_pool = []
    for bid in val_book2label.keys():
        for vol_id, chk in chunk_by_volume(bid, TXT_VAL_DIR):
            val_pool.append((chk, val_book2label[bid], bid, vol_id))
         
    #外部验证集
    train_txt, train_lbl, _, train_vid = zip(*train_pool)
    val_txt,   val_lbl, _, _   = zip(*val_pool)

    # 分词 & 数据集
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    train_ds = build_dataset(train_txt, train_lbl)  
    val_ds   = build_dataset(val_txt, val_lbl)

    #val_ds = val_ds.shuffle(seed=42)        # <-  打乱验证集
    #val_ds   = build_dataset([], [])          # 空验证集，Trainer 需要

    # 权重采样（新增加）
    
    # 1. 先保留 chunk→label 的映射，但把权重算到“chunk 级别”
    train_vid = []                      # 复用你前面代码
    for bid in book2label.keys():
        for vol_id, _ in chunk_by_volume(bid, TXT_TRAIN_DIR):
            train_vid.append(vol_id)

    

    # 2. 重新遍历一次，把“每个窗口”对应到它所属 chunk 的权重
    vol2nwins = prescan_windows()       # 第 1 步预统计结果

    win_weights = []
    for text, _, bid, vol_id in train_pool:
        nwins = len(slide_window(text))
        if nwins == 0:
            continue
        win_weights.extend([vol_weight[vol_id] / vol2nwins[vol_id]] * nwins)

    win_weights = torch.tensor(win_weights, dtype=torch.float)
    win_weights = win_weights * (len(win_weights) / win_weights.sum())
    assert len(win_weights) == len(train_ds), \
        f"权重数 {len(win_weights)} ≠ 窗口数 {len(train_ds)}"
    # 3. 现在 win_weights 长度 == 窗口总数，可以直接喂 WeightedRandomSampler
    


    # 模型 & 训练参数
    

    config = BertConfig.from_pretrained(model_dir)
    config.num_labels = 2
    config.problem_type = "single_label_classification"   # ← 加这里
    config.hidden_dropout_prob = 0.1

    model = BertForSequenceClassification.from_pretrained(
        model_dir, config=config)

    #1/4epoch保存一次

    n_windows = len(train_ds)                      # 总窗口数
    bsz       = 8                                 #batch_size
    steps_per_epoch = n_windows // bsz
    save_steps      = max(1, steps_per_epoch // 4)  # 1/4 epoch                   

    args = TrainingArguments(
        output_dir=CKPT_ROOT,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,                 
        learning_rate=2e-5,
        weight_decay=1e-6,
        
        eval_strategy="steps",              
        eval_steps=save_steps,              
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,        
        metric_for_best_model="auc",
        logging_steps=save_steps,
        fp16=True,
        warmup_ratio=0.02,
        dataloader_drop_last=True,
        lr_scheduler_type="cosine" , #尾部曲线调整
    )

    def compute_metrics(pred):
        logits, labels = pred
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        auc = roc_auc_score(labels, probs)
        # ---- 自定义落盘 ----
        pd.DataFrame({'prob': probs, 'label': labels}).to_csv(
            r'train\csv\result\val_pred_last_epoch.csv', index=False) 
        return {'auc': auc}    

    trainer = Trainer(    #老训练器
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        
    )
    #trainer = WeightedTrainer(
    #    train_sampler_weights=win_weights,   # ← 把权重传进来
    #    model=model,
    #    args=args,
    #    train_dataset=train_ds,
    #    eval_dataset=val_ds,
    #    tokenizer=tokenizer,
    #    data_collator=DataCollatorWithPadding(tokenizer),
    #    compute_metrics=compute_metrics,
    #)

    trainer.add_callback(  #加入早停
        EarlyStoppingCallback(
            early_stopping_patience=3,   
            early_stopping_threshold=0.0005  
        )
    )


    print('>>> CUDA 可用:', torch.cuda.is_available())
    print('>>> 使用设备:', trainer.args.device)
    if not torch.cuda.is_available():
        print("CUDA 不可用，程序终止。")
        exit()

    print('>>> 开始训练（1 epoch，约 2.5h）...')
    trainer.train()
    tokenizer.save_pretrained(CKPT_ROOT)
    model.save_pretrained(CKPT_ROOT)
    print('>>> 训练完成，模型保存至', CKPT_ROOT)