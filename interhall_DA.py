
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[26]:


df_X = pd.read_csv("train_upd.csv")


# In[27]:


df_X.head()


# In[8]:


df_test = pd.read_csv("test_upd.csv")


# In[9]:


df_test.head()


# In[45]:


nlist = [i for i in range(1,38)]


# In[73]:


X = df_X.iloc[:,nlist].values


# In[74]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[75]:


labelencoder_X = LabelEncoder()


# In[76]:


X[:,36] = labelencoder_X.fit_transform(X[:,36])


# In[77]:


onehotencoder = OneHotEncoder(categorical_features = [36])


# In[78]:


X = onehotencoder.fit_transform(X).toarray()


# In[79]:


X = X[:,:-1]


# In[81]:


X


# In[84]:


labelencoder_y = LabelEncoder()


# In[82]:


y = df_X.iloc[:,38].values


# In[85]:


y = labelencoder_y.fit_transform(y)


# In[86]:


y


# In[87]:


from sklearn.cross_validation import train_test_split


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)


# In[89]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[90]:


X_train = sc.fit_transform(X_train)


# In[91]:


X_test = sc.transform(X_test)


# In[92]:


from sklearn.ensemble import RandomForestClassifier


# In[123]:


from sklearn.model_selection import GridSearchCV


# In[126]:


parameters = [{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}]


# In[128]:


grid_search = GridSearchCV(estimator = classifier, param_grid= parameters, scoring='accuracy', cv = 10)


# In[ ]:


grid_search.fit(X_train,y_train)
grid_search.best_params_


# In[ ]:


classifier = RandomForestClassifier(n_estimators = 1200, criterion = 'entropy', random_state = 0)


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


from sklearn.metrics import matthews_corrcoef


# In[ ]:


matthews_corrcoef(y_test, y_pred)  


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, y_pred)

