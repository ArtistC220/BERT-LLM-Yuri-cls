import pandas as pd
from utils import load_config
config = load_config()
import os

path_df = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","weighted","weighted.csv")
df = pd.read_csv(path_df)  # 原文件
df['book_id'] = df['text_id'].str.split('_').str[0]   # 提取书号
book_df = df.groupby('book_id')['weighted'].mean().reset_index()

path_bookdf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","weighted","book_weighted.csv")
book_df.to_csv(path_bookdf, index=False, encoding='utf-8-sig')

print('已生成 book_weighted.csv')