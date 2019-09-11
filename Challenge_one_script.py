
# coding: utf-8


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
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_predict , cross_val_score ,KFold
import math
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
#get_ipython().magic(u'matplotlib inline')



df = pd.read_csv("SubCh1_TrainingData.csv")

df.set_index("Sample_Name",inplace=True)


df_edit = pd.concat( [df, pd.get_dummies(df["Treatment"]),pd.get_dummies( df["Timepoint"])],axis=1 )


df_edit= df_edit.drop(["Treatment","Timepoint","BioRep","Isolate"],axis=1)


df = df_edit


y = df["DHA_IC50"]


X = df.drop("DHA_IC50",axis=1)

scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)

df_test = pd.read_csv("SubCh1_TestData.csv")

df_test_edit = pd.concat( [df_test, pd.get_dummies(df_test["Treatment"]),pd.get_dummies( df_test["Timepoint"])],axis=1 )

df_test_edit=df_test_edit.drop(["Treatment","Timepoint","BioRep","Isolate"],axis=1)

df_test = df_test_edit


#df_test.set_index("Sample_Names",inplace=True)
#X_test = df_test.drop("DHA_IC50",axis=1)
#y_test = df_test["DHA_IC50"]
X_train, X_test, y_train, y_test = train_test_split(scaled_data,y,test_size=0.33, random_state=101)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 5000, stop = 7000, num = 10)]
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


# In[17]:

base_model = RandomForestRegressor(n_estimators = 10, random_state = 101)
base_model.fit(X_train, y_train)



# In[ ]:


cv = KFold(10, shuffle=True)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = base_model, param_distributions = random_grid, n_iter = 100, cv = cv, verbose=2, random_state=101, n_jobs = -1)


# In[ ]:


rf_random.fit(X_train,y_train )


# In[ ]:


print (rf_random.best_params_)


# In[ ]:


# y_pred=rf_random.predict(X_test)


# In[ ]:


# print (classification_report(y_test, y_pred))


# In[ ]:


# print ("MCC %f"%(matthews_corrcoef(y_test, y_pred)))
# print ("Accuracy %f"%(accuracy_score(y_test, y_pred)))


# In[18]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[19]:




# In[20]:


base_accuracy = evaluate(base_model, X_test, y_test)


# In[21]:


predictions = base_model.predict(X_test)


# In[22]:

print ("base model")
print ("R^2:",r2_score(y_test, predictions))
print ("MAE:",mean_absolute_error(y_test, predictions))
print ("MSE:",mean_squared_error(y_test, predictions))


# In[ ]:


best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test )

predictions = rf_random.predict(X_test)
print ("best RFR")
print ("R^2:",r2_score(y_test, predictions))
print ("MAE:",mean_absolute_error(y_test, predictions))
print ("MSE:",mean_squared_error(y_test, predictions))

# In[ ]:


print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
