#%%
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")
#%%
train_data.shape
test_data.shape
train_data.columns
#%%
training_data = train_data.drop(["Col1","Col2"],axis=1)
testing_data = test_data.drop("Col1",axis=1)
target = train_data.Col2
ID = test_data.Col1
#%%
target.value_counts()
combined_data = pd.concat([training_data,testing_data],axis=0)
#%%
colmn = combined_data.columns
for i in range(combined_data.shape[1]):
  if combined_data[colmn[i]].isnull().sum() > 2000:
    combined_data.drop(colmn[i],axis=1,inplace=True)
#%%
from sklearn.preprocessing import LabelEncoder
leb = LabelEncoder()
col = combined_data.columns
for i in combined_data.columns:
	if combined_data[i].dtype == "int64" or combined_data[i].dtype == "float64":
		combined_data[i].fillna(np.max(combined_data[i]),inplace=True)
	elif combined_data[i].dtype == "object":
		combined_data[i].fillna(np.max(combined_data[i]),inplace=True)
		leb.fit_transform(combined_data[i],inplace=True)
#%%
combined_data.fillna(combined_data.median())
#%%
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import scale
combined_data = scale(combined_data)

train_ddata = combined_data[0:17521,:]
test_ddata = combined_data[17521::,:]
#%%
train_ddata = pd.DataFrame(train_ddata)
train_ddata = pd.concat([train_ddata,target],axis=1)
train_ddata.shape
#%%
Q1 = train_ddata.quantile(0.1)
Q3 = train_ddata.quantile(0.99)
IQR = Q3 - Q1
#%%
train_ddata1 = train_ddata[~((train_ddata < (Q1 - 1.5 * IQR)) |(train_ddata > (Q3 + 1.5 * IQR))).any(axis=1)]
train_ddata1.shape
train_ddata2 = train_ddata1.iloc[:,0:1635]
targett = train_ddata1.iloc[:,1635]
#%%
sm = ADASYN()
X,Y = sm.fit_sample(train_ddata2,targett)
from sklearn.utils import shuffle
for i in range(200):
    X,Y = shuffle(X,Y,random_state=0)
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,stratify=Y,random_state=42,test_size=0.2)
#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,make_scorer
score = make_scorer(f1_score)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
clf3 = GaussianNB()
rnd_clf = RandomForestClassifier(max_depth = 3,n_estimators=300,random_state=42,n_jobs=-1,class_weight={0:0.8,1:1.4})
extree_tree_clf = ExtraTreesClassifier(max_depth = 3,n_estimators=300,random_state=42,n_jobs=-1,class_weight={0:0.8,1:5})
ada_clf = AdaBoostClassifier(extree_tree_clf,n_estimators= 100,algorithm="SAMME.R",learning_rate=0.05)
mlp_clf = MLPClassifier(random_state=42)
log_clf = LogisticRegression(class_weight={0:0.8,1:1})
model_LGBM =lgb.LGBMClassifier(learning_rate=0.01,objective='binary')

from sklearn.ensemble import VotingClassifier
named_estimators = [("rnd_clf",rnd_clf),("extree_tree_clf",extree_tree_clf),("log_clf",log_clf)]
voting_clf = VotingClassifier(named_estimators,voting="soft",flatten_transform=True)
voting_clf.fit(X_train,y_train)
result = voting_clf.predict(X_test)
f_score = f1_score(y_test,y_pred)
print ("f_score ",f_score)
#%%
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(max_depth = 9,n_estimators=500,random_state=42,n_jobs=-1,class_weight={0:0.9,1:5})
from sklearn.metrics import accuracy_score,f1_score
model = rnd_clf.fit(X_train,y_train)
y_pred = model.predict(X_test)
f_score = f1_score(y_test,y_pred)
print ("f_score ",f_score)
#%%
X.shape
test_ddata.shape
#%%
rnd_clf = RandomForestClassifier(max_depth = 10,n_estimators=500,random_state=42,n_jobs=-1,class_weight={0:0.8,1:3})
rnd_clf.fit(X,Y)
predict = rnd_clf.predict(test_ddata)
#%%
test_result = pd.concat([ID,pd.DataFrame(predict)],axis=1)
test_result.columns = ["Col1","Col2"]
submit=test_result.to_csv("hdfc_loan1.csv", encoding='utf-8', index=False)
#%%
