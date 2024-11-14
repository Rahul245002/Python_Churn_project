#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from imblearn.combine import SMOTEENN


# In[2]:


dt=pd.read_csv('customer_churn.csv')
dt.head()


# In[3]:


dt.head()


# In[4]:


dt=dt.drop('Unnamed: 0', axis=1)


# In[5]:


dt.head()


# In[6]:


x=dt.drop(['Churn'], axis=1)
x


# In[7]:


y=dt['Churn']
y


# In[8]:


x_train, x_test, y_train, y_test=train_test_split(x,y, stratify=y, test_size=0.2, random_state=100)


# In[9]:


print(x_train.shape, y_train.shape, x_test.shape)


# # Logistic Regression

# In[10]:


model_1=LogisticRegression()
model_1.fit(x_train, y_train)


# In[11]:


y_pred= model_1.predict(x_test)
y_pred


# In[12]:


model_1.score(x_test,y_test)


# In[13]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[14]:


sm=SMOTEENN()
x_resampled, y_resampled = sm.fit_resample(x,y)


# In[15]:


xr_train, xr_test, yr_train, yr_test = train_test_split(x_resampled, y_resampled, test_size=0.2)


# In[16]:


model_2_sm= LogisticRegression()
model_2_sm.fit(xr_train, yr_train)


# In[17]:


y_pred_sm=model_2_sm.predict(xr_test)
model_2_score=model_2_sm.score(xr_test, yr_test)
print(model_2_score)
print(classification_report(yr_test, y_pred_sm))


# In[18]:


print(confusion_matrix(yr_test, y_pred_sm))


# In[19]:


print((492+607)/(492+46+49+607)*100)


# # Decision Tree

# In[20]:


model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[21]:


model_dt.fit(x_train, y_train)


# In[22]:


y_pred_dt=model_dt.predict(x_test)
y_pred_dt


# In[23]:


model_dt.score(x_test, y_test)


# In[24]:


print(classification_report(y_test, y_pred))


# In[25]:


sm=SMOTEENN()
x_resampled1, y_resampled1=sm.fit_resample(x,y)


# In[26]:


xd_train, xd_test, yd_train, yd_test=train_test_split(x_resampled1, y_resampled1, test_size=0.2)


# In[27]:


model_dt_smote=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[28]:


model_dt_smote.fit(xd_train, yd_train)
yd_pred_sm=model_dt_smote.predict(xd_test)
model_d_score=model_dt_smote.score(xd_test, yd_test)
print(model_d_score)
print(classification_report(yd_test, yd_pred_sm))


# # Random Forest

# In[29]:


model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[30]:


model_rf.fit(x_train, y_train)


# In[31]:


yrf_pred=model_rf.predict(x_test)


# In[32]:


model_rf.score(x_test, y_test)


# In[33]:


print(classification_report(y_test, yrf_pred))


# In[34]:


sm=SMOTEENN()
x_resampled2, y_resampled3= sm.fit_resample(x,y)


# In[35]:


xrf_train, xrf_test, yrf_train, yrf_test=train_test_split(x_resampled2, y_resampled3,test_size=0.2)


# In[36]:


model_rf_smoten=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model_rf_smoten.fit(xrf_train, yrf_train)


# In[37]:


y_rf_pred=model_rf_smoten.predict(xrf_test)


# In[38]:


model_rf_score=model_rf_smoten.score(xrf_test,yrf_test)


# In[39]:


print(model_rf_score)
print(classification_report(yrf_test, y_rf_pred))


# In[40]:


print(confusion_matrix(yrf_test, y_rf_pred))


# In[41]:


print((476+633)/(476+57+28+633)*100)


# 
# Random Forest model is more accurate compare to other models. so finalize RF Classifier, and save the model so that we can use it in a later stage

# In[42]:


import pickle


# In[43]:


filename='Model.sav'


# In[44]:


pickle.dump(model_rf_smoten, open(filename, 'wb'))


# In[45]:


load_model=pickle.load(open(filename, 'rb'))


# In[46]:


model_score_r1=load_model.score(xrf_test, yrf_test)


# In[47]:


model_score_r1


# Our final model i.e. RF Classifier with SMOTEENN, is now ready and dumped in model.sav, which we will use and prepare API's so that we can access our model from UI.

# In[ ]:




