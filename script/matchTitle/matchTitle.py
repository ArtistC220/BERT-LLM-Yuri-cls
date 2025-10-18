import pandas as pd
import re

# 读取文件
df1 = pd.read_csv(r'train\csv\result02\大模型\volume_weighted_large.csv', encoding='utf-8')
df2 = pd.read_csv(r'train\script\matchTitle\baihe_aid_title_cover.csv', encoding='gbk')

# 从 filename 中提取 bookid（数字部分在第一个下划线前）
#df1['book_id'] = df1['filename'].str.extract(r'^(\d+)_')[0].astype(int)

# 重命名 aid 为 bookid，便于合并
df2.rename(columns={'aid': 'book_id'}, inplace=True)

# 只保留需要的列
df2_subset = df2[['book_id','title']]

# 合并
df_merged = pd.merge(df1, df2_subset, on='book_id', how='left')

# 保存结果
df_merged.to_csv(r'train\csv\result02\大模型\volume_weighted_with_title_large.csv', index=False, encoding='utf-8-sig')