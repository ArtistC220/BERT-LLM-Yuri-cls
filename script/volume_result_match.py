import pandas as pd
import os
from utils import load_config
config = load_config()

def read_csv_safe(path, **kwargs):
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"❌ 无法读取文件: {path}, 尝试的编码都失败了")

# 1. 读取 weighted.csv
path_dfw = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "csv", "weighted", "weighted.csv"
)
df_weight = read_csv_safe(path_dfw)

# 2. 读取 filename_realvolume.csv
path_dfv = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "csv", "result", "filename_realvolume.csv"
)
df_volume = read_csv_safe(path_dfv)

# 3. 读取 baihe_aid_title_cover_local.csv
path_dft = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "csv", "BaiHe_aid_title_cover", "baihe_aid_title_cover_local.csv"
)
df_title = read_csv_safe(path_dft)

# 4. 从 filename 中提取 aid 和 vid
df_weight[['aid', 'vid']] = df_weight['text_id'].str.split('_', expand=True)
df_volume[['aid', 'vid']] = df_volume['text_id'].str.split('_', expand=True)

# 5. 转换 aid 为整数类型（如果有 NaN，先 dropna 或填充）
df_weight['aid'] = df_weight['aid'].astype(int)
df_volume['aid'] = df_volume['aid'].astype(int)
df_title['aid'] = df_title['aid'].astype(int)

# 6. 合并
merged = df_weight.merge(df_volume[['aid', 'vid', 'realvolume']], on=['aid', 'vid'], how='left')
merged = merged.merge(df_title[['aid', 'title']], on='aid', how='left')

# 7. 选择所需列并排序
result = merged[['aid', 'vid', 'title', 'realvolume', 'weighted']].copy()
result = result.sort_values(by=['aid', 'realvolume'])

# 8. 保存结果
path_out = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "csv", "result", "volume_result.csv"
)
result.to_csv(path_out, index=False, encoding='utf-8-sig')

print("✅ 合并完成，已输出 volume_result.csv")
