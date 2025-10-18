import pandas as pd
import re
import os
from utils import load_config
config = load_config()

# 1. 读入 3 个文件
path_df1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","prediction","model_BERT_prediction.csv")
path_df2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","normalized","new_dialogue_normalized.csv")
path_df3 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","normalized","new_verb_normalized.csv")


df1 = pd.read_csv(path_df1)
df2 = pd.read_csv(path_df2)
df3 = pd.read_csv(path_df3)

# 2. 去掉 .txt，生成纯 filename 列
df1['text_id'] = df1['filename']

# 3. 按 filename 对齐（inner join，确保三边都出现）
merged = (df1[['text_id', 'pred_prob']]
          .merge(df2[['text_id', 'yuri_norm']], on='text_id', how='inner')
          .merge(df3[['text_id', 'yuri_norm']], on='text_id', how='inner',
                 suffixes=('_line', '_actpsy'))) #按filename合并三个文件中需要的数据


# 4. 加权平均
merged['weighted'] = (0.6 * merged['pred_prob'] +
                      0.3 * merged['yuri_norm_line'] +
                      0.1 * merged['yuri_norm_actpsy'])  #权值可以调整

# 5. 保存结果
path_1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","weighted","weighted_detial.csv")
path_2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","weighted","weighted.csv")

merged.to_csv(path_1, index=False, encoding='utf-8-sig') #带细节版本
out = merged[['text_id', 'weighted']]
out.to_csv(path_2, index=False, encoding='utf-8-sig')

print('Done!')