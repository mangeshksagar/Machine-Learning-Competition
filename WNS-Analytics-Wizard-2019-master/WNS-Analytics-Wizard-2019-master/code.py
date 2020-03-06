#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#%%
train = pd.read_csv("train.csv")
#train2 = pd.read_csv("item_data.csv")
#train3 = pd.read_csv("log.csv")
test = pd.read_csv("test.csv")
#%%
train.columns
test.columns
train.isnull().sum()
test.isnull().sum()
train.dtypes
test.dtypes
#%%
from sklearn.preprocessing import LabelEncoder
lm = LabelEncoder()
a  = ['os_version']
for i in np.arange(len(a)):
    train[a[i]] = lm.fit_transform(train[a[i]])
    
for i in np.arange(len(a)):
    test[a[i]] = lm.fit_transform(test[a[i]])
#%%

train['impression_time'] = pd.to_datetime(train.impression_time,format='%Y-%m-%d %H:%M') 
test['impression_time'] = pd.to_datetime(test.impression_time,format='%Y-%m-%d %H:%M') 

for i in (train, test):
    i['year']=i.impression_time.dt.year 
    i['month']=i.impression_time.dt.month 
    i['day']=i.impression_time.dt.day
    i['Hour']=i.impression_time.dt.hour 
    i['Minutes']=i.impression_time.dt.minute

#%%
ID = test["impression_id"]
y = train['is_click']

X = train.drop(['impression_id','impression_time','is_click'],axis=1)
Test = test.drop(['impression_id','impression_time'],axis=1)
#%%
#split the traindata into train and test
#X_train, X_test, y_train, y_test = train_test_split( X,y , test_size=0.2, random_state=42)
#
##%%
#from sklearn.metrics import f1_score
#model = RandomForestClassifier(random_state=42,max_depth= 10,min_samples_split=5 ,n_estimators=500,n_jobs=-1,min_samples_leaf=2,criterion='gini')
#model.fit(X_train,y_train)
#y_pred = model.predict(X_test)
#acc = model.score(X_test,y_test)
#F1 = f1_score(y_test,y_test)
#%%
#from imblearn.under_sampling import AllKNN
#allknn = AllKNN()
#X, y = allknn.fit_resample(X, y)
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X, y = sm.fit_sample(X, y)
#from imblearn.under_sampling import TomekLinks
#Tm = TomekLinks()
#X, y = Tm.fit_resample(X, y)
#from imblearn.over_sampling import ADASYN 
#sm = ADASYN()
#X, y = sm.fit_sample(X, y)
#X = pd.DataFrame(X, columns = X.columns)
#%%
model = RandomForestClassifier(random_state=42,max_depth= 10,min_samples_split=5 ,n_estimators=500,n_jobs=-1,min_samples_leaf=2, criterion='entropy')#,class_weight= {0:1,1:10})
model.fit(X,y)
y_pred = model.predict(Test)
print(sum(y_pred))
#%%,eval_metric="F1"Logloss ,l2_leaf_reg=3
from catboost import CatBoostClassifier
model = CatBoostClassifier( iterations =1000,depth=7,learning_rate=0.01,task_type="GPU",devices='0:1',eval_metric="F1",border_count=254)
model.fit(X,y,plot=False)
y_pred = model.predict(Test)
#%%
from sklearn.linear_model import LogisticRegression 
#from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(X, y)
y_pred = model.predict(Test)
#%%
import xgboost as xgb
model = xgb.XGBClassifier( learning_rate = 0.01,max_depth = 7, n_estimators = 500)

model.fit(X,y)

y_pred = model.predict(Test)
#%%
model = CatBoostClassifier(iterations=1000,task_type="GPU",devices='0:1')
model.fit(X,y,plot=False)
y_pred = model.predict(Test)
#%%
pd.DataFrame({'impression_id':ID,'is_click': y_pred}).set_index('impression_id').to_csv('3.csv')
pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)

#%%