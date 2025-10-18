import pandas as pd

# 1. 读入刚才上传的文件
df = pd.read_csv(r'train\csv\result02\val_vol_mean.csv')

# 2. 生成 filename：bookid_volumeid（vol_id 补 4 位）
df['filename'] = df['book_id'].astype(str) + '_' + df['vol_id'].astype(str).str.zfill(4)

# 3. 只要加权脚本需要的两列，并重命名
out = df[['filename', 'prob']].rename(columns={'prob': 'pred_prob'})

# 4. 保存
out.to_csv(r'train\csv\result02\val_vol_mean_for_weight.csv',
           index=False, encoding='utf-8-sig')

print('val_vol_mean 已转成 filename+pred_prob 格式')

#如果正在使用的卷级概率没有filename请使用这个....