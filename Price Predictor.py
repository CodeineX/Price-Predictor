#!/usr/bin/env python
# coding: utf-8

# # Price Predictor for an Estate 
# ## Took from Boston Housing Data

#     Features in the housing data
#     
#     1. CRIM      per capita crime rate by town
#     2. ZN        proportion of residential land zoned for lots over 
#                  25,000 sq.ft.
#     3. INDUS     proportion of non-retail business acres per town
#     4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#                  river; 0 otherwise)
#     5. NOX       nitric oxides concentration (parts per 10 million)
#     6. RM        average number of rooms per dwelling
#     7. AGE       proportion of owner-occupied units built 
#                  prior to 1940
#     8. DIS       weighted distances to five Boston employment centres
#     9. RAD       index of accessibility to radial highways
#     10. TAX      full-value property-tax rate per 10,000 USD
#     11. PTRATIO  pupil-teacher ratio by town
#     12. B        1000(Bk - 0.63)^2 where Bk is the proportion of 
#                  blacks by town
#     13. LSTAT    % lower status of the population
#     14. MEDV     Median value of owner-occupied homes in 1000's USD

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing = pd.read_csv("data.csv") #housing DataFrame


# In[3]:


housing.head()


# In[4]:


housing['CHAS'].value_counts()


# Note: This field is rather interesting because while creating training and teting data set, it may so happen that all entries with 'CHAS' value = 0 is present in the training dataset and our model does not know that the value can also be 1 for some entries.

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


housing.hist(bins = 50, figsize = (20,15))
plt.show() #All features in the form of a histogram


# ## Train-Test Splitting

# ### Using scikit-learn

# In[8]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 0)


# Using StratifiedShuffleSplit based on entries in 'CHAS' so that entries with values 0 and 1 get distributed uniformly between training and testing dataset.

# In[9]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# ## Looking for correlations

# In[10]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False) 
#This shows correlation of all the attributes with respect to MEDV (which is the median value of housing price)


# In[11]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (20,15))
plt.show()
#This shows correlation of each of attributes with each other


# In[12]:


housing.plot(kind="scatter", x="RM", y="MEDV")
plt.show()


# ## Trying Out Attribute Combinations

# In[13]:


housing["TAXRM"] = housing["TAX"]/ housing["RM"]


# In[14]:


housing.head()


# In[15]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False) 
#This shows correlation of all the attributes with respect to MEDV (which is the median value of housing price)


# In[16]:


#Surprisingly, we got a new attribute with a high negative correlation


# In[17]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV")
plt.show()


# ### Separating features and label

# In[18]:


housing = strat_train_set.drop("MEDV", axis = 1)
housing_label = strat_train_set["MEDV"].copy()


# ## Taking care of missing attributes

# In[19]:


housing = strat_train_set.copy()


# In[20]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[21]:


X =  imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns = housing.columns)
#transformed dataframe 


# # Creating Pipeline

# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
#    ....add as many as you need in your pipeline
    ('std_scaler', StandardScaler())
])


# In[23]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# ## Selecting a desired model

# In[24]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()


# In[25]:


model.fit(housing_num_tr, housing_label)


# In[26]:


some_data = housing.iloc[:5]
some_label = housing_label.iloc[:5]


# In[27]:


prepared_data = my_pipeline.transform(some_data)


# In[28]:


model.predict(prepared_data)


# In[29]:


list(some_label)


# ## Evaluating the model

# In[30]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_label, housing_predictions)
rmse = np.sqrt(mse)


# In[31]:


mse


# ## Cross Validation Technique

# In[32]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_label, scoring="neg_mean_squared_error", cv=10)
rmse_scores  =np.sqrt(-scores)


# In[33]:


def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Std Deviation: ", scores.std())


# In[34]:


print_scores(rmse_scores)


# In[35]:


from joblib import dump, load
dump(model, "PricePredictor.joblib")


# ## Testing Data

# In[36]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test  =strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[ ]:




