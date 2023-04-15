import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
def data(path,feature_num):
    data = pd.read_excel(path)
    Data = np.array(data)
    x = Data[:, :feature_num]
    y = Data[:, feature_num:]
    return x,y

def Max(y_test,y_pred):
    y_test_max = np.max(y_test)
    y_pred_max = np.max(y_pred)
    if y_test_max>y_pred_max:
        return (y_test_max+y_test_max*0.01)
    else:
        return (y_pred_max+y_pred_max*0.01)

def Min(y_test,y_pred):
    y_test_min=np.min(y_test)
    y_pred_min = np.min(y_pred)
    if y_pred_min<y_test_min:
        return (y_pred_min-y_pred_min*0.01)
    else:
        return (y_test_min-y_test_min*0.01)


def draw(y_test,y_pred,score,title,colors):
    '''
    :param y_test: Actual values
    :param y_pred: Predicted values
    :param score: R^2 value
    :param title: Title of the plot and filename of saved image (in the format title.png)
    :param colors: Color of the points
    :return: None
    '''
    plt.figure()
    plt.title(title,family='Arial', fontsize=20)
    # Add text (coordinates can be modified)
    plt.text(0.05, 0.2, 'RMSE:{}\nR^2:{}'.format(round(math.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2), round(score, 2)), family='Times New Roman', fontsize=15)
    # Add points, alpha is the transparency of the points
    plt.scatter(y_test, y_pred,c=colors,alpha=0.5)
    # Define the x and y axis labels
    plt.ylabel('Prediction', family='Arial', weight='normal', fontsize=15)
    plt.xlabel('Observation', family='Arial', weight='normal', fontsize=15)
    plt.xticks( family='Arial', weight='normal', fontsize=13)
    plt.yticks(family='Arial', weight='normal', fontsize=13)
    # Set the x and y limits of the plot
    y_max=Max(y_test,y_pred)
    y_min=Min(y_test,y_pred)
    plt.xlim(y_min,y_max)
    plt.ylim(y_min,y_max)
    # Draw a diagonal line, modify c= to change color, and modify (1,0) and (1,0) to modify the line parameters
    plt.plot((1, 0), (1, 0), transform=plt.gca().transAxes, ls='-', c='#DC143C', label="1:1 line")
    plt.plot((1, 0), (1.1, 0.1), transform=plt.gca().transAxes, ls='--', c='#DC143C', label="1:1 line")
    plt.plot((1.1, 0.1), (1, 0), transform=plt.gca().transAxes, ls='--', c='#DC143C', label="1:1 line")
    plt.savefig(title+'.pdf',dpi=600)
    plt.show()


if name=="main":
# The first parameter is the input file name, and the second parameter is the number of features
    x,y=data("training_data/data_for_train_r.xlsx",6)
# Load the model, modify the name to load different models
    model = joblib.load("model/model_T_total_0322.pkl")
# Use the model to obtain different predicted values
    y_pred=model.predict(x)
# Obtain the R^2 value
    score=sklearn.metrics.r2_score(y, y_pred)
# Plot the graph, see the function for details
    draw(y,y_pred,score,'model_T_total_0322','#00CED1')

