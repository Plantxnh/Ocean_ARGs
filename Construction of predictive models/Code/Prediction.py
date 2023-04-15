import joblib
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl


path = "/Users/xnh/Library/CloudStorage/OneDrive-个人/11全球海洋ARG/00投稿/Code/Construction of predictive models/Input"
outputpath = "output"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
path_list=os.listdir(path)
for filename in path_list:
    file=path+"/"+filename
    data = pd.read_excel(file)  #reading data
    print(data)
    Data=np.array(data)

    data_ML=np.stack(Data)[:,3:9]
    data_info=np.stack(Data)[:,0:3]

    forest=joblib.load("Model/Demo.pkl")
    predict=forest.predict(data_ML)

    Data_new=np.c_[data_info,predict]
    new_Data=pd.DataFrame(Data_new)
    new_Data.columns = ["year","lat","lon", "pred"]
    print(new_Data)

    outfile=outputpath+"/"+filename
    new_Data.to_excel(outfile,index=False)


