import pandas as pd
import joblib

def robust_minmax_transform(df: pd.DataFrame) -> pd.DataFrame:
    rob = joblib.load(r'train\script\normalizer\actpsy_robust_step.pkl')
    mm  = joblib.load(r'train\script\normalizer\actpsy_mm_step.pkl')
    tmp = rob.transform(df[['yuri_concentration']])
    df['yuri_norm'] = mm.transform(tmp)
    return df

# 用法
new = pd.read_csv('new_file.csv', usecols=['filename', 'yuri_concentration']) #将需要归一化的新的csv数据相对路径放到这里
new = robust_minmax_transform(new)
new[['filename', 'yuri_norm']].to_csv(r'train\csv\normalizerCSV\new_line_normalized.csv',
                                      index=False, encoding='utf-8-sig')