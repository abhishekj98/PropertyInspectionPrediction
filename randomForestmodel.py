
# In[80]:


# Liberty Mutual Group: Property Inspection Prediction
import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib


# In[81]:


#load train and test 
df  = pd.read_csv('train.csv', index_col=0)
tf  = pd.read_csv('test.csv', index_col=0)


# In[82]:


df.head()


# In[83]:


tf.head()


# In[84]:


tf.drop('T2_V10', axis=1, inplace=True)
tf.drop('T2_V7', axis=1, inplace=True)
tf.drop('T1_V13', axis=1, inplace=True)
tf.drop('T1_V10', axis=1, inplace=True)


# In[85]:


tf.shape


# In[86]:


labels = df.Hazard
df.drop('Hazard', axis=1, inplace=True)


# In[87]:


df.drop('T2_V10', axis=1, inplace=True)
df.drop('T2_V7', axis=1, inplace=True)
df.drop('T1_V13', axis=1, inplace=True)
df.drop('T1_V10', axis=1, inplace=True)


# In[88]:


df.shape


# In[89]:


df.head()


# In[90]:


columns = df.columns
test_ind = tf.index


# In[91]:


print(columns)
print(test_ind)


# In[92]:


df = np.array(df)
tf = np.array(tf)


# In[93]:


print(df)
print(tf)


# In[94]:


# label encode the categorical variables
for i in range(df.shape[1]):
    if type(df[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[:,i]) + list(tf[:,i]))
        df[:,i] = lbl.transform(df[:,i])
        tf[:,i] = lbl.transform(tf[:,i])

df = df.astype(float)
tf = tf.astype(float)


# In[95]:


print(df)
print(tf)


# In[96]:


param_grid = {'n_estimators': [100]}
model = GridSearchCV(RandomForestRegressor(), param_grid)


# In[97]:


model = model.fit(df,labels)
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)


# In[98]:


preds = model.predict(tf)
joblib.dump(model, 'Random_forest_regressor_model.pkl')


# In[99]:


#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('abhishek_jagtap_rf_model.csv')

