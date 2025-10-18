import pandas as pd

df = pd.read_csv(r'train\csv\result02\大模型\volume_weighted_large.csv')  # 原文件
#df['book_id'] = df['filename'].replace(r'_\d+' , '' , regex= True)  #被注释这一段是针对只有filename没有book_id的csv准备的
book_df = df.groupby('book_id')['weighted'].mean().reset_index()
book_df.to_csv(r'train\csv\result02\大模型\book_weighted_large.csv', index=False, encoding='utf-8-sig')



print('done')


#这是一个临时手动卷级概率合并到书级概率的脚本
#若要手动合并窗口概率到卷级概率，请使用check_vol_agg.py