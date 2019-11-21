
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set(style="white")


# In[4]:


import numpy as np


# In[5]:


import pandas as pd


# In[6]:


df_X = pd.read_csv("train_upd.csv")


# In[7]:


df_X = df_X.drop(columns = ['Unnamed: 39','Unnamed: 40'])


# In[8]:


df_X.head()


# In[9]:


df_X1 = df_X.drop(columns = ['par_year', 'par_month', 'par_hour','par_min','cell_range','tilt','ran_vendor','Congestion_Type'])


# In[10]:


df_X1.columns.values 


# In[11]:


df_X1 = df_X1.drop(columns=[' Cell_name'])


# In[7]:


X=df_X.iloc[:,1:].drop(columns=['par_year','par_month'])


for index,row in X.iterrows():
    X['par_min']=X['par_hour']+X['par_min']/60
X=X.drop(columns=['par_hour'])
X['par_day']=X['par_day']%7


# In[8]:


y=df_X['Congestion_Type'].values


# In[9]:


X = X.drop(columns = ['Congestion_Type'])


# In[10]:


X = X.values


# In[11]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[12]:


labelencoder_X = LabelEncoder()


# In[13]:


X[:,33] = labelencoder_X.fit_transform(X[:,33])


# In[14]:


onehotencoder = OneHotEncoder(categorical_features = [33])


# In[15]:


X = onehotencoder.fit_transform(X).toarray()


# In[16]:


X[:,35] = labelencoder_X.fit_transform(X[:,35])


# In[17]:


X[:,34] = labelencoder_X.fit_transform(X[:,34])


# In[18]:


X[:,4] =labelencoder_X.fit_transform(X[:,4])


# In[12]:


df_X1.columns


# In[ ]:


cols = ['4G_rat', 'par_day', 'subscriber_count', 'web_browsing_total_bytes',
       'video_total_bytes', 'social_ntwrking_bytes',
       'cloud_computing_total_bytes', 'web_security_total_bytes',
       'gaming_total_bytes', 'health_total_bytes', 'communication_total_bytes',
       'file_sharing_total_bytes', 'remote_access_total_bytes',
       'photo_sharing_total_bytes', 'software_dwnld_total_bytes',
       'marketplace_total_bytes', 'storage_services_total_bytes',
       'audio_total_bytes', 'location_services_total_bytes',
       'presence_total_bytes', 'advertisement_total_bytes',
       'system_total_bytes', 'voip_total_bytes', 'speedtest_total_bytes',
       'email_total_bytes', 'weather_total_bytes', 'media_total_bytes',
       'mms_total_bytes', 'others_total_bytes', 'beam_direction',
       'Total_bytes', '1/cell_range_square', 'cell_range_square',
       'inner_radius', 'subscriber/r2', 'Total_bytes/r2', 'rin-rout',
       'subscriber/(rin-rout)^2', '(rin-rout)^2', 'total_byte/(rin-rout)^2']
pp = sns.pairplot(df_X1[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)


# In[20]:


f, ax = plt.subplots(figsize=(10, 6))
corr = wines.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)

