import pandas as pd

# 1. 读入
df = pd.read_csv(r'train/csv/result02/val_vol_prob_large.csv')

# 2. 定义统一的替换规则（正则模式）
pat = r'tensor\((\d+)\)'      # 匹配 tensor(数字)
repl = r'\1'                 # 保留数字

# 3. 对指定三列做正则替换
cols = ['filename', 'book_id', 'vol_id']
df[cols] = df[cols].replace(to_replace=pat, value=repl, regex=True)

# 4. 写出
df.to_csv(r'train/csv/result02/val_vol_prob_large_clean.csv', index=False)

print('clean done -> val_vol_prob_large_clean.csv')