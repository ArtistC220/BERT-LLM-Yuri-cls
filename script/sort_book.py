import pandas as pd
from utils import load_config
# 读取配置
import os

path_df = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","result","book_result.csv")
df = pd.read_csv(path_df)     # 输入文件
df = df.sort_values('weighted', ascending=False)  # 降序
path_dfout = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv","result","book_result_sorted.csv")
df.to_csv(path_dfout, index=False, encoding='utf-8-sig')  # 输出文件

print('已降序排序')