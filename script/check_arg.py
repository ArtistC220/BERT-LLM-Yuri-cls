#!/usr/bin/env python3
"""
show_args.py
查看 Transformers 训练时真正的超参（weight_decay、warmup_ratio 等）
用法：
    python show_args.py train\models\ckpt6\checkpoint-47200
"""
import sys
import torch
import transformers
from pathlib import Path

CKPT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('train/models/ckpt6/checkpoint-47200')
ARGS_FILE = CKPT_DIR / 'training_args.bin'

if not ARGS_FILE.exists():
    print(f' 找不到文件：{ARGS_FILE}')
    sys.exit(1)

# 放行 Transformers 自定义类
torch.serialization.add_safe_globals([transformers.TrainingArguments])

args = torch.load(ARGS_FILE, weights_only=False)

# 只挑常用关键超参
keys = ['weight_decay', 'warmup_ratio', 'learning_rate', 'lr_scheduler_type',
        'per_device_train_batch_size', 'num_train_epochs', 'seed']
print(f'  checkpoint 路径：{CKPT_DIR.resolve()}\n')
for k in keys:
    print(f'{k:30s}: {getattr(args, k)}')