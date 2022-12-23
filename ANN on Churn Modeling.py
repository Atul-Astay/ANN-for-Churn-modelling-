#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("C:\\Users\\astay\\Downloads\\Churn_modelling.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[69]:


sns.countplot(x = 'Geography', data = data)
#plt.hist(x = 'Geography',data = data)


# In[74]:


sns.countplot(x = 'Exited',data = data)


# In[5]:


data.tail()


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


y = data.iloc[:,13].values
X = data.iloc[:,3:13].values


# In[9]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
one_hot = OneHotEncoder()
label_encode = LabelEncoder()


# In[10]:


data['Gender'] = label_encode.fit_transform(data['Gender'])


# In[11]:


data.Gender.value_counts()


# In[12]:


data['Geography'] = label_encode.fit_transform(data["Geography"])


# In[13]:


data['Geography'].value_counts()


# In[20]:


y = data.iloc[:,13].values
X = data.iloc[:,3:13].values


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3,random_state = 0)


# In[22]:


from sklearn.preprocessing import StandardScaler
s_scale = StandardScaler()
X_train = s_scale.fit_transform(X_train)
X_test = s_scale.transform(X_test)


# In[24]:


X_train


# In[25]:


X_test


# # Building ANN

# In[26]:


import tensorflow as tf


# In[28]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[29]:


classifier = Sequential()


# In[32]:


classifier.add(Dense(units=64, kernel_initializer='uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))


# In[33]:


classifier.compile(optimizer='adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


# In[35]:


history = classifier.fit(X_train,y_train,epochs = 100, batch_size = 64)


# # Prediction

# In[49]:


y_pred = classifier.predict(X_test)


# In[37]:


y_pred


# In[51]:


y_pred = y_pred>0.5
y_pred


# In[52]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
cm


# In[53]:


accuracy_score(y_test,y_pred)


# In[54]:


for i in range(0,y_pred.size):
    if y_pred[i] > 0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0


# In[58]:


sns.heatmap(cm,annot = True,fmt ='.0f')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of Churn Modelling")
plt.show()


# In[ ]:




