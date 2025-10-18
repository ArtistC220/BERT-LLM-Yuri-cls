import pandas as pd
import os
from utils import load_config

config = load_config()

def read_csv_safe(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件: {path}, 尝试的编码都失败了")

# 1. 读取加权结果
path_dfw = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "csv", "weighted", "book_weighted.csv"
)
df_w = read_csv_safe(path_dfw)
print("df_w 列:", df_w.columns.tolist())

if "book_id" in df_w.columns:
    df_w = df_w.rename(columns={'book_id': 'aid'})
elif "aid" not in df_w.columns:
    raise KeyError(" book_weighted.csv 里没有 'book_id' 或 'aid' 列")

# 2. 读取标题映射表
path_dft = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "csv", "BaiHe_aid_title_cover", "baihe_aid_title_cover_local.csv"
)
df_t = read_csv_safe(path_dft)
print("df_t 列:", df_t.columns.tolist())

if "aid" not in df_t.columns:
    raise KeyError("baihe_aid_title_cover_local.csv 里没有 'aid' 列")

df_t = df_t[['aid', 'title']]

# 3. 类型统一为字符串，避免 object vs int64 报错
df_w['aid'] = df_w['aid'].astype(str)
df_t['aid'] = df_t['aid'].astype(str)

# 4. 合并
df_out = df_w.merge(df_t, on='aid', how='left')

# 5. 保存结果
path_dfout = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "csv", "result", "book_result.csv"
)
df_out.to_csv(path_dfout, index=False, encoding='utf-8-sig')

print(' 匹配完成，已输出 book_result.csv')
