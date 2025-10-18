from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pandas as pd, joblib, os

# 1. 旧数据第一次运行
old = pd.read_csv(r'train\csv\Action_psy_count_satistic.csv',
                  usecols=['text_id', 'yuri_concentration'])
old.rename(columns={'text_id': 'filename'}, inplace=True) 

rob = RobustScaler()          # 先去掉极端值
mm  = MinMaxScaler(feature_range=(0, 1))  # 再压到 0-1

tmp = rob.fit_transform(old[['yuri_concentration']])
old['yuri_norm'] = mm.fit_transform(tmp)

old[['filename', 'yuri_norm']].to_csv(
    r'train\csv\normalizerCSV\actpsy_old_normalized.csv',
    index=False, encoding='utf-8-sig')

# 存两套参数
joblib.dump(rob, r'train\script\normalizer\actpsy_robust_step.pkl')
joblib.dump(mm,  r'train\script\normalizer\actpsy_mm_step.pkl')

joblib.dump(rob, r'train\script\normalizer\actpsy_robust_step4.pkl', protocol=4)
joblib.dump(mm,  r'train\script\normalizer\actpsy_mm_step4.pkl',  protocol=4)