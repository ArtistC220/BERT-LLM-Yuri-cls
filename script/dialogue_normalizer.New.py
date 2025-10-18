import pandas as pd
import joblib
import os
from utils import load_config
config = load_config()

def robust_minmax_transform(df: pd.DataFrame) -> pd.DataFrame:
    path_rob = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["models_dir"],"./dialogue_robust_step.pkl")
    rob = joblib.load(path_rob)
    path_mm = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["models_dir"],"./dialogue_mm_step.pkl")
    mm  = joblib.load(path_mm)
    tmp = rob.transform(df[['yuri_concentration']])
    df['yuri_norm'] = mm.transform(tmp)
    return df

# 用法
path_new = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_prediction_dir"],"./LLM_dialogue_prediction.csv")
new = pd.read_csv(path_new, usecols=['text_id', 'yuri_concentration']) #将需要归一化的新的csv数据相对路径放到这里
new = robust_minmax_transform(new)
path_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_dir"],"./normalized/new_dialogue_normalized.csv")
new[['text_id', 'yuri_norm']].to_csv(path_csv,index=False, encoding='utf-8-sig')