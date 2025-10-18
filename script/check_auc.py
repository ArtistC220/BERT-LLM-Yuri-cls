import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
df = pd.read_csv(r'train\csv\result02\val_window_prob.csv')
print('AUC =', roc_auc_score(df.label, df.prob))
print('ACC =', accuracy_score(df.label, df.prob > 0.5))
print('logloss =', log_loss(df.label, df.prob))