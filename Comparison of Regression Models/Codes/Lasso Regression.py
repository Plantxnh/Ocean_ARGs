import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn.linear_model import Lasso
import joblib

path = "training_data/data_for_train_r.xlsx"

k = 10
data = pd.read_excel(path)
name = list(data.columns.values)
Data = np.array(data)
np.random.shuffle(Data)
num_val_sample = len(Data) // k
feature_num = 6
print(Data.shape)
score_list = []
best_lasso = Lasso(alpha=1.0)
best_score = 0
best_score_list = []

for i in range(k):
    x_test = Data[i * num_val_sample:(i + 1) * num_val_sample, 0:feature_num]
    y_test = Data[i * num_val_sample:(i + 1) * num_val_sample, feature_num:]
    x_train = np.concatenate([Data[:i * num_val_sample, 0:feature_num], Data[(i + 1) * num_val_sample:, 0:feature_num]],
                             axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]],
                             axis=0)

    lasso = Lasso(alpha=1.0)
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    score = sklearn.metrics.r2_score(y_test, y_pred)
    if best_score < score:
        best_score = score
        best_lasso = lasso
    score_list.append(score)
print(score_list)
print(sum(score_list) / len(score_list))
joblib.dump(best_lasso, "model/LR_model_T_total_0322.pkl")

print(best_score)
for i in range(k):
    x_test = Data[i * num_val_sample:(i + 1) * num_val_sample, 0:feature_num]
    y_test = Data[i * num_val_sample:(i + 1) * num_val_sample, feature_num:]
    x_train = np.concatenate([Data[:i * num_val_sample, 0:feature_num], Data[(i + 1) * num_val_sample:, 0:feature_num]],
                             axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]],
                             axis=0)
    y_pred = best_lasso.predict(x_test)
    score = sklearn.metrics.r2_score(y_test, y_pred)
    best_score_list.append(score)
print(best_score_list)
print(sum(best_score_list) / len(best_score_list))
score_all = []
for i in range(Data.shape[1]):
    score_all.append(best_score_list)
plt.figure()
plt.boxplot(score_all, labels=name)
plt.show()


