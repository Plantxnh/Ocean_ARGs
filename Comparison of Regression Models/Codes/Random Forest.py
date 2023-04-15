import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# Display settings
# mpl.rcParams['font.sans-serif'] = [u'SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

path = "/Demo_data/demo.xlsx"

# K-fold cross-validation
k = 10
data = pd.read_excel(path)  # Read data
name = list(data.columns.values)
Data = np.array(data)
np.random.shuffle(Data)
num_val_sample = len(Data) // k
feature_num = 6
print(Data.shape)
score_list = []

# Handling missing values
for i in range(k):
    x_test = Data[i * num_val_sample:(i + 1) * num_val_sample, 0:feature_num]
    y_test = Data[i * num_val_sample:(i + 1) * num_val_sample, feature_num:].ravel()
    x_train = np.concatenate([Data[:i * num_val_sample, 0:feature_num], Data[(i + 1) * num_val_sample:, 0:feature_num]],
                             axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]],
                             axis=0).ravel()

    # Using GridSearchCV to find the best parameters
    param_grid = {'n_estimators': [10, 25, 50, 75, 100], 'max_depth': [None, 5, 10, 15],
                  'min_samples_split': [2, 5, 10]}
    forest = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=k, scoring='r2')
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    best_forest = grid_search.best_estimator_

    y_pred = best_forest.predict(x_test)
    score = sklearn.metrics.r2_score(y_test, y_pred)
    score_list.append(score)

print(score_list)
print(sum(score_list) / len(score_list))
joblib.dump(best_forest, "Model/demo.pkl")

importances = best_forest.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(feature_num):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Get the best solution
print(grid_search.best_score_)
best_score_list = []
for i in range(k):
    x_test = Data[i * num_val_sample:(i + 1) * num_val_sample, 0:feature_num]
    y_test = Data[i * num_val_sample:(i + 1) * num_val_sample, feature_num:].ravel()
    x_train = np.concatenate([Data[:i * num_val_sample, 0:feature_num], Data[(i + 1) * num_val_sample:, 0:feature_num]],
                             axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]],
                             axis=0).ravel()

    y_pred = best_forest.predict(x_test)
    score = sklearn.metrics.r2_score(y_test, y_pred)
    best_score_list.append(score)

print(best_score_list)
print(sum(best_score_list) / len(best_score_list))

# Plot feature importance
importances = best_forest.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(feature_num):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(feature_num), importances[indices],
       color="r", align="center")
plt.xticks(range(feature_num), np.array(name)[indices])
plt.xlim([-1, feature_num])
plt.show()
