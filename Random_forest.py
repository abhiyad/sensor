
# coding: utf-8

# In[9]:


import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# In[60]:


f=open('data1.txt')
a=[]
y=[]
x=[]
for line in f :
    a.append(line.replace('\n','\t').split('\t')[0:4])
for i in a:
    y.append(i[0])
    x.append(float(i[1]))
    x.append(float(i[2]))
    x.append(float(i[3]))
X=np.array(x)
Y=np.array(y)
X=X.reshape(-1,3) # back to 2D representation


# In[61]:


lb = preprocessing.LabelBinarizer()
lb.fit(Y)
lb.classes_
Y=lb.transform(Y) # Y is now a one Hot vector
Y_new = np.zeros(Y.shape[0])
for j in range(Y.shape[0]):
    Y_new[j] = int(np.argmax(Y[j]))
Y=Y_new
X=preprocessing.normalize(X)


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


# In[63]:


ran = RandomForestClassifier(max_depth=20, random_state=0)
ran.fit(X_train,y_train)


# In[64]:


y_pred = ran.predict(X_train)
count = 0 
for j in range (y_train.shape[0]):
    if y_train[j] == y_pred[j]:
        count = count + 1
print "Training Accuracy :" ,count*100.0/y_train.shape[0]


# In[65]:


y_pred = ran.predict(X_test)
count = 0 
for j in range (y_test.shape[0]):
    if y_test[j] == y_pred[j]:
        count = count + 1
print "Test Accuracy : ",count*100.0/y_test.shape[0]

