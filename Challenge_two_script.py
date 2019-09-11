
# coding: utf-8

# In[1]:


import os ,sys 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from operator import itemgetter
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,classification_report, confusion_matrix,accuracy_score,matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.svm import SVC, SVR
from scipy.stats import zscore
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_predict , cross_val_score ,KFold
import math
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[2]:


df = pd.read_csv("SubCh2_TrainingData.csv")


# In[ ]:


# for m in df.columns:
#     print (m)


# In[ ]:


# df["Country"].unique()


# In[ ]:


# df["Asexual.stage..hpi."].unique()


# In[ ]:


# df["Kmeans.Grp"].unique()


# In[ ]:


# df["ClearanceRate"].unique()


# In[ ]:


# df[df["ClearanceRate"].isna()]


# In[ ]:


# for m in df.columns:
#     print (m,df[df[m].isna()].shape[0])


# In[3]:


df.set_index("Sample_Names",inplace=True)


# In[ ]:


# df_2["Country"].unique()


# In[ ]:


# for m in df_2.columns:
#     print (m)


# In[4]:


df.dropna(subset=["ClearanceRate"],inplace=True)


# In[5]:


X = df.drop(["ClearanceRate","Country","Kmeans.Grp"],axis=1)
y = df["ClearanceRate"]


# In[7]:


for m in X.columns :
    X[m].fillna((X[m].mean()),inplace=True)
    #sub2['income'].fillna((sub2['income'].mean()), inplace=True)


# In[8]:


X_noNan = pd.concat( [X, pd.get_dummies( df["Kmeans.Grp"]  )],axis=1,sort=True)


# In[10]:


# for m in X_train.columns:
#     print (m,X_train[X_train[m].isna()].shape[0])
print (X_noNan.shape)
print (y.shape)


# In[12]:


rus = RandomUnderSampler(random_state=101)


# In[14]:


X_resample , y_resample = rus.fit_sample(X_noNan, y)


# In[ ]:


# y_train = pd.get_dummies(y) 


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X_resample , y_resample, test_size=0.33, random_state=101)


# In[16]:


base_model = RandomForestClassifier(n_estimators = 1000, random_state = 101)


# In[18]:


base_model.fit(X_train, y_train)


# In[19]:


y_pred = base_model.predict(X_test)


# In[20]:


print ("base models ")
print (classification_report(y_test,y_pred))


# In[28]:


print ("MCC")
print (matthews_corrcoef(y_test,y_pred))


# In[22]:


print ("Accuracy")
print (accuracy_score(y_test,y_pred))


# In[23]:


y.value_counts()


# In[24]:


print (X_train.shape)
print (y_train.shape)


# In[ ]:


# df_2 = pd.read_csv("SubCh2_TestData.csv")


# In[25]:


# y_pred


# In[26]:


### coeficient for prediction 
y_coef = base_model.predict_proba(X_test)


# In[27]:


# y_coef


# In[29]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 2000, stop = 7000, num = 30)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 220, num = 22)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8 ]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[30]:


cv = KFold(10, shuffle=True)


# In[31]:


rf_random = RandomizedSearchCV(estimator = base_model, param_distributions = random_grid, n_iter = 100, cv = cv, verbose=2, random_state=101, n_jobs = -1)


# In[ ]:


print ("hyper param ")
rf_random.fit(X_train,y_train )
print (rf_random.best_params_)
y_pred = rf_random.predict(X_test)

print (classification_report(y_test,y_pred))
print ("Accuracy")
print (accuracy_score(y_test,y_pred))
print ("MCC")
print (accuracy_score(y_test,y_pred))

