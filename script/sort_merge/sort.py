import pandas as pd

df = pd.read_csv(r'train\csv\result02\大模型\volume_weighted_with_title_large.csv')     # 输入文件
df = df.sort_values('weighted', ascending=False)  # 降序
df.to_csv(r'train\csv\result02\大模型\volume_weighted_with_title_sorted_large.csv', index=False, encoding='utf-8-sig')  # 输出文件

print('已降序排序')

#这是一个临时用于手动排序的脚本