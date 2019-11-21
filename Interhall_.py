
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import xgboost


# In[3]:


import pandas as pd


# In[4]:


from sklearn.cross_validation import *


# In[5]:


df_X = pd.read_csv("train_upd.csv")


# In[6]:


df_X = df_X.drop(columns = ['Unnamed: 39','Unnamed: 40'])


# In[7]:


df_X.shape


# In[8]:


X=df_X.iloc[:,1:].drop(columns=['par_year','par_month'])


for index,row in X.iterrows():
    X['par_min']=X['par_hour']+X['par_min']/60
X=X.drop(columns=['par_hour'])
X['par_day']=X['par_day']%7


# In[9]:


y=df_X['Congestion_Type'].values


# In[10]:


X = X.drop(columns = ['Congestion_Type'])


# In[11]:


X = X.values


# In[12]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[13]:


labelencoder_X = LabelEncoder()


# In[14]:


X[:,33] = labelencoder_X.fit_transform(X[:,33])


# In[15]:


onehotencoder = OneHotEncoder(categorical_features = [33])


# In[16]:


X = onehotencoder.fit_transform(X).toarray()


# In[17]:


X[:,34] = labelencoder_X.fit_transform(X[:,34])


# In[18]:


X[:,35] = labelencoder_X.fit_transform(X[:,35])


# In[19]:


X[:,4] =labelencoder_X.fit_transform(X[:,4])


# In[20]:


labelencoder_y = LabelEncoder()


# In[21]:


y = labelencoder_y.fit_transform(y)


# In[22]:


from sklearn.cross_validation import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)


# In[24]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


# In[40]:


#clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)


# In[27]:


#clf2 = RandomForestClassifier(n_estimators=1000, random_state=1)


# In[28]:


#clf3 = GaussianNB()


# In[41]:


#eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')


# In[32]:


from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import matthews_corrcoef
mcc = make_scorer(matthews_corrcoef)


# In[ ]:


#clf1 = DecisionTreeClassifier(max_depth=4)
#clf2 = KNeighborsClassifier(n_neighbors=7)
#clf3 = SVC(gamma='scale', kernel='rbf', probability=True)
#>>> eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
#...                         voting='soft', weights=[2, 1, 2])


# In[53]:


#clf1 = clf1.fit(X_train, y_train)
#clf2 = clf2.fit(X_train, y_train)
#clf3 = clf3.fit(X_train, y_train)
#eclf = eclf.fit(X_train, y_train)


# In[61]:


#eclf = eclf.fit(X_train, y_train)
#clf2 = clf2.fit(X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[62]:


param = {'C': 0.7678243129497218, 'penalty': 'l1'}
model1 = LogisticRegression(**param)

param = {'n_neighbors': 15}
model2 = KNeighborsClassifier(**param)

param = {'C': 1.7, 'kernel': 'linear'}
model3 = SVC(**param)

param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
model4 = DecisionTreeClassifier(**param)

param = {'learning_rate': 0.05, 'n_estimators': 1500}
model5 = AdaBoostClassifier(**param)

param = {'learning_rate': 0.01, 'n_estimators': 1000}
model6 = GradientBoostingClassifier(**param)

model7 = GaussianNB()

model8 = RandomForestClassifier()

model9 = ExtraTreesClassifier()


# In[63]:


estimators = [('LR',model1), ('KNN',model2), ('SVC',model3),
              ('DT',model4), ('ADa',model5), ('GB',model6),
              ('NB',model7), ('RF',model8),  ('ET',model9)]


# In[67]:


from sklearn.model_selection import StratifiedKFold 


# In[ ]:


kfold = StratifiedKFold(n_splits=10, random_state=0)
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X_train,y_train, cv=kfold)
print('Accuracy on train: ',results.mean())
ensemble_model = ensemble.fit(X_train,y_train)
y_pred = ensemble_model.predict(X_test)
print('Accuracy on test:',(y_test == pred).mean())


# In[51]:


matthews_corrcoef(y_test, y_pred)  

