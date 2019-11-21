# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 00:39:34 2019
poly features
np.mean(res)
Out[172]: 0.7879242666403897---mattcoeff

np.mean(history.history['val_acc'])
Out[173]: 0.7744727587103843

np.mean(history.history['acc'])
Out[174]: 0.8135617162251806
@author: amrit
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('train_upd.csv')
# =============================================================================
# #--------------------------------------TESTING DATA-----------------------------------
# 
# #-------------------------------------------------------------------------------------
# =============================================================================
test_data=pd.read_csv('test_upd.csv')
X=data.iloc[:,1:-1].drop(columns=['par_year','par_month'])

y=pd.DataFrame(data.iloc[:,-1])
for index,row in X.iterrows():
    X['par_min']=X['par_hour']+X['par_min']/60
X=X.drop(columns=['par_hour'])
X['par_day']=X['par_day']%7
X_num=X.iloc[:,:-1]
# =============================================================================
# a=pd.DataFrame(np.zeros(shape=len(X_num)).reshape(78560,1),columns=['square range'])
# b=pd.DataFrame(np.zeros(shape=len(X_num)).reshape(78560,1),columns=['subscriber per squared range'])
# a=X_num['cell_range']**2
# b=X_num['subscriber_count']/(X_num['cell_range']**2)
# col=X_num.columns
# X_num=np.c_[X_num,a,b]
# X_num=pd.DataFrame(X_num,columns=list(col)+['square range','subscriber per squared range'])
# =============================================================================


X_cat=pd.DataFrame(X.iloc[:,[1,-3,-2,-1]])
X1=X_num['4G_rat']
X_num=X_num.drop(columns=['4G_rat'])


from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2)
X_num=poly.fit_transform(X_num)

X_num=np.c_[X1,X_num]

X_num=np.delete(X_num,[1],axis=1)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from	sklearn.base	import	BaseEstimator,	TransformerMixin

# =============================================================================
# class	DataFrameSelector(BaseEstimator,	TransformerMixin):
# 				def	__init__(self,	attribute_names):
# 								self.attribute_names	=	attribute_names
# 				def	fit(self,	X,	y=None):
# 								return	self
# 				def	transform(self,	X):
# 								return	X[self.attribute_names].values
# =============================================================================
                       
# =============================================================================
#                             
# '''num_attribs=list(X_num)
# cat_attribs=["ran_vendor"]
# num_pipeline=Pipeline([('std_scaler',StandardScaler),])
# cat_pipeline	=Pipeline([('selector',	DataFrameSelector(cat_attribs)),('label_binarizer',	LabelBinarizer()),])
# from sklearn.pipeline import FeatureUnion
# full_pipeline=FeatureUnion(transformer_list=[('num_pipeline',num_pipeline),('cat_pipeline',cat_pipeline),])  
# 
# X_prepared=num_pipeline.fit_transform(np.array(X_num))
# '''
# =============================================================================
std_scaler=StandardScaler()
X_prepared_num=std_scaler.fit_transform(X_num)

#X_test_prepared_num=std_scaler.transform(X_test_num)

encoder=LabelBinarizer()
X_prepared_cat=encoder.fit_transform(X_cat['ran_vendor'])
X_prepared=np.c_[X_prepared_num,X_prepared_cat]
X_prepared_cat=encoder.fit_transform(X_cat['tilt'])
X_prepared=np.c_[X_prepared,X_prepared_cat]
X_prepared_cat=encoder.fit_transform(X_cat['cell_range'])
X_prepared=np.c_[X_prepared,X_prepared_cat]
X_prepared_cat=encoder.fit_transform(X_cat['par_day'])
X_prepared=np.c_[X_prepared,X_prepared_cat]
#X_test_prepared_cat=encoder.transform(X_test_cat)
#X_test_prepared=np.c_[X_test_prepared_num,X_test_prepared_cat]




#X_prepared=X_prepared.T;
#y_prepared=y_prepared.T;


'''from	sklearn.ensemble	import	 RandomForestClassifier
sgd_clf	=RandomForestClassifier(random_state=42,n_estimators=100,bootstrap=False)
sgd_clf.fit(X_prepared,	y)

from	sklearn.model_selection	import	cross_val_score
cross_val_score(sgd_clf,	X_prepared,	y,	cv=2,	scoring="accuracy")'''
for index,row in y.iterrows():
    if row['Congestion_Type']=='NC':
        row['Congestion_Type']=0
    elif row['Congestion_Type']== '4G_RAN_CONGESTION':
        row['Congestion_Type']=1
    elif row['Congestion_Type']== '4G_BACKHAUL_CONGESTION':
        row['Congestion_Type']=2
    elif row['Congestion_Type']== '3G_BACKHAUL_CONGESTION':
        row['Congestion_Type']=3  
        
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.05, random_state=42)
        
from tensorflow.keras import regularizers
model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',),
     keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.001),
                kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',),
   

   keras.layers.Dense(4, activation=tf.nn.softmax)
])
                    
from tensorflow.keras import backend as K



#def matthews_correlation(y_true, y_pred):
# return matthews_corrcoef(y_true,y_pred)
# =============================================================================
# def matthews_correlation(y_true, y_pred):
#     ''' Matthews correlation coefficient
#     '''
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pred_neg = 1 - y_pred_pos
# 
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     y_neg = 1 - y_pos
# 
#     tp = K.sum(y_pos * y_pred_pos)
#     tn = K.sum(y_neg * y_pred_neg)
# 
#     fp = K.sum(y_neg * y_pred_pos)
#     fn = K.sum(y_pos * y_pred_neg)
# 
#     numerator = (tp * tn - fp * fn)
#     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
# 
#     return numerator / (denominator + K.epsilon())
#     
# =================================COMPILING NEURAL NET============================================
from tensorflow.keras.optimizers import Adam
adam=Adam(lr=0.001,decay=1e-6)
model.compile(optimizer=adam,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  
# =============================================================================
# for index,row in y_test.iterrows():
#     if row['Congestion_Type']=='NC':
#         row['Congestion_Type']=0
#     elif row['Congestion_Type']== '4G_RAN_CONGESTION':
#         row['Congestion_Type']=1
#     elif row['Congestion_Type']== '4G_BACKHAUL_CONGESTION':
#         row['Congestion_Type']=2
#     elif row['Congestion_Type']== '3G_BACKHAUL_CONGESTION':
#         row['Congestion_Type']=3         
# 
# =============================================================================
##the model has already been trained use the trained parameters from loaded model       
model.fit(X_prepared, np.array(y.iloc[:,0]),validation_split=0.1 ,epochs=200,batch_size=1000)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from sklearn.metrics import matthews_corrcoef

y_pred1=model.predict_classes(X_test)
res=matthews_corrcoef(y_test.astype(int64),y_pred1)





from tensorflow.keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model1(2).json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1(2).h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model1(2).json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1(2).h5")
print("Loaded model from disk")

# =============================================================================
# loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# score_on_train=loaded_model.evaluate(X_train, np.array(y_train.iloc[:,0]), verbose=0)
# print("train accuracy",score_on_train[1])
# # =============================================================================
# # #score_on_test = loaded_model.evaluate(X_test_prepared, np.array(y_test.iloc[:,0]), verbose=0)
# # #print("test accuracy",score_on_test[1])
# # 
# # #from	sklearn.model_selection	import	cross_val_score
# # #cross_val_score(loaded_model,X_prepared,np.array(y.iloc[:,0]),	cv=3,	scoring="accuracy")
# # =============================================================================
# =============================================================================


#-----------------------------------------------------------------------------------------------



