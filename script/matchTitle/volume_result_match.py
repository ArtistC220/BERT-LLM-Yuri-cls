import pandas as pd

df_weight = pd.read_csv(r'train\csv\result\volume_weighted.csv')


df_volume = pd.read_csv(r'train\csv\result\filename_realvolume.csv')


df_title = pd.read_csv(r'train\csv\baihe_aid_title_cover\baihe_aid_title_cover.csv', encoding='gbk')

# 从 filename 中提取 aid 和 vid
df_weight[['aid', 'vid']] = df_weight['filename'].str.split('_', expand=True)
df_volume[['aid', 'vid']] = df_volume['filename'].str.split('_', expand=True)

# 转换 aid 为整数类型，便于匹配
df_weight['aid'] = df_weight['aid'].astype(int)
df_volume['aid'] = df_volume['aid'].astype(int)
df_title['aid'] = df_title['aid'].astype(int)

# 合并三个表
merged = df_weight.merge(df_volume[['aid', 'vid', 'realvolume']], on=['aid', 'vid'], how='left')
merged = merged.merge(df_title[['aid', 'title']], on='aid', how='left')

# 选择所需列并保留 vid
result = merged[['aid', 'vid', 'title', 'realvolume', 'weighted']].copy()

# 排序：先按 aid，再按卷序（realvolume 字符串排序，如需自然序可再优化）
result = result.sort_values(by=['aid', 'realvolume'])

# 保存
result.to_csv(r'train\csv\result\volume_result.csv', index=False, encoding='utf-8-sig')
print("合并完成")