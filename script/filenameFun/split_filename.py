import pandas as pd
df = pd.read_csv(r'train\csv\result02\大模型\volume_weighted_with_title_sorted_large.csv')
# 假设 filename 形如 "100_0001.txt"
df[['aid','vid']] = df['filename'].str.replace('.txt','',regex=False).str.split('_',expand=True)
df.to_csv(r'train\csv\history_rank_volume\history_rank_volume_large.csv', index=False, encoding='utf-8-sig')