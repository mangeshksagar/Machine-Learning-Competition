#%%
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")
#%%
train_data.shape
test_data.shape
train_data.columns
#%%
target = train_data.Col2
ID = test_data["Col1"]
training_data = train_data.drop(['Unnamed: 0',"Col1","Col2"],axis=1)
testing_data = test_data.drop(['Unnamed: 0',"Col1"],axis=1)

#%%
aa=target.value_counts()
combined_data = pd.concat([training_data,testing_data],axis=0)
#%%
A = combined_data.isnull().sum()
mk = list(np.where(combined_data.dtypes==object)[0])
null = combined_data.columns[mk]
combined_data = combined_data.drop(null,axis=1)
#%%
from sklearn.preprocessing import scale
combined_data = scale(combined_data)

train_ddata = combined_data[0:17521,:]
test_ddata = combined_data[17521::,:]
#%%^
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_ddata,target ,stratify=target ,random_state=42,test_size=0.15)

#%%
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(max_depth = 10,n_estimators=500,random_state=42,n_jobs=-1,class_weight={0:0.8,1:1.2})
from sklearn.metrics import accuracy_score,f1_score
model = rnd_clf.fit(X_train,y_train)
y_pred = model.predict(X_test)
f_score = f1_score(y_test,y_pred)
print ("f_score ",f_score)
#%%
rnd_clf = RandomForestClassifier(max_depth = 9,n_estimators=600,random_state=42,n_jobs=-1,class_weight={0:0.9,1:5})
rnd_clf.fit(train_ddata,target)
predict = rnd_clf.predict(test_ddata)
#%%
test_result = pd.concat([ID,pd.DataFrame(predict)],axis=1)
test_result.columns = ["Col1","Col2"]
submit=test_result.to_csv("hdfc_loan1.csv", encoding='utf-8', index=False)
#%%
