import pandas as pd

# 1. 读取
df_w = pd.read_csv(r'train\csv\result02\大模型\book_weighted_large.csv')\
        .rename(columns={'book_id': 'aid'})          # 把 bookid 改成 aid
df_t = pd.read_csv(r'train\script\matchTitle\baihe_aid_title_cover.csv',
                   encoding='gbk', usecols=['aid','title'])

# 2. 合并
df_out = df_w.merge(df_t, on='aid', how='left')

# 3. 保存
df_out.to_csv(r'train\csv\result02\大模型\book_weighted_with_title_large.csv',
              index=False, encoding='utf-8-sig')

print('匹配完成，已输出 book_result.csv')