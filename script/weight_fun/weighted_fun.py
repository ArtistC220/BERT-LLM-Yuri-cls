import pandas as pd
import re
import os


# 1. 读入 3 个文件
df1 = pd.read_csv(r'train\csv\result02\大模型\val_vol_prob_large.csv')# 含 .txt
df2 = pd.read_csv(r'train\csv\normalizerCSV\line_old_normalized.csv')
df3 = pd.read_csv(r'train\csv\normalizerCSV\actpsy_old_normalized.csv')

# 2. 去掉 .txt，生成纯 filename 列
df1['filename'] = df1['filename'].str.replace(r'\.txt$', '', regex=True)

# 3. 按 filename 对齐（inner join，确保三边都出现）
merged = (df1[['filename', 'pred_prob']]
          .merge(df2[['filename', 'yuri_norm']], on='filename', how='inner')
          .merge(df3[['filename', 'yuri_norm']], on='filename', how='inner',
                 suffixes=('_line', '_actpsy'))) #按filename合并三个文件中需要的数据


# 4. 加权平均
merged['weighted'] = (0.6 * merged['pred_prob'] +
                      0.3 * merged['yuri_norm_line'] +
                      0.1 * merged['yuri_norm_actpsy'])  #权值可以调整

# 5. 保存结果
merged.to_csv(r'train\csv\result02\大模型\volume_weighted_detial_large.csv', index=False, encoding='utf-8-sig') #带细节版本
out = merged[['filename', 'weighted']]
out.to_csv(r'train\csv\result02\大模型\volume_weighted_large.csv', index=False, encoding='utf-8-sig')

print('Done')


#这是一个手动做最后加权的脚本
#更改目录请确保csv中的标签名字一致


