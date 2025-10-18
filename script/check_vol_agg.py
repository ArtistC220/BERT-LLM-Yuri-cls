import pandas as pd

#若要手动把窗口概率合并为卷级概率，请用这个。

# 1. 路径
IN_CSV  = r'train/csv/result02/val_vol_prob_large.csv'   # 现在文件的位置
OUT_CSV = r'train/csv/result02/val_vol_prob_large.csv'  # 输出“一卷一行”

# 2. 读入
df = pd.read_csv(IN_CSV)

# 4. 每卷一行：平均概率 + 保留标签，并生成 filename
vol_mean = (
    df.groupby(['book_id', 'vol_id'], as_index=False)
      .agg(pred_prob=('pred_prob', 'mean'),          # 改名，方便后续加权脚本
           label=('label', 'first'))
      .assign(
           filename=lambda x: x['book_id'].astype(str) + '_' + x['vol_id'].astype(str).str.zfill(4)
       )
      # 调整列顺序
      [['filename', 'book_id', 'vol_id', 'pred_prob', 'label']]
)

# 5. 保存
vol_mean.to_csv(OUT_CSV, index=False, encoding='utf-8')

# 6. 自查
print('原始窗口行数:', len(df))
print('合并后卷数  :', len(vol_mean))
print(vol_mean.head())