#!/usr/bin/env python
# coding: utf-8

# Importing Required Liberaries:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    


# Loading the file:

# In[2]:


df_train = pd.read_excel(r'C:\Users\DELL\Documents\Data_Train (2).xlsx')
df_test = pd.read_excel(r'C:\Users\DELL\Documents\Data_Test (2).xlsx')
df_train.head()


# In[3]:


df_train['Name'] = df_train.Name.str.split().str.get(0)
df_train.head()


# In[4]:


df_test['Name'] = df_test.Name.str.split().str.get(0)
df_test.head()


# Null Removal:

# In[5]:


df_train.isnull().sum()


# In[6]:


df_test.isnull().sum()


# In[7]:


df_train['Seats'].fillna(df_train['Seats'].mean(),inplace=True)


# In[8]:


df_test['Seats'].fillna(df_test['Seats'].mean(),inplace=True)


# In[9]:


df_train.isnull().sum()


# In[10]:


df_test.isnull().sum()


# Since Mileage is not integer type , we cannot use the mean() meathod here.
# Therefore:
# we will create a graph and see the higher values and replace the null values with them.

# In[11]:


data = pd.concat([df_train,df_test], sort=False)   #WE combime the datasets for ease of use...


# In[12]:


plt.figure(figsize=(20,5))
data['Mileage'].value_counts().head(100).plot.bar()
plt.show()


# In[13]:


df_train['Mileage'] = df_train['Mileage'].fillna('17.0 kmpl')
df_test['Mileage'] = df_test['Mileage'].fillna('17.0 kmpl')

# Notice that in the above graph the values some values are 0 as well.
#Just to be on the safe side :

df_train['Mileage'] = df_train['Mileage'].replace("0.0 kmpl", "17.0 kmpl")
df_test['Mileage'] = df_test['Mileage'].replace("0.0 kmpl", "17.0 kmpl")


# Now we do the same for every Column that has null values:

# In[14]:


plt.figure(figsize=(20,5))
data['Engine'].value_counts().head(100).plot.bar()
plt.show()


# In[15]:


df_train['Engine'] = df_train['Engine'].fillna('1197 CC')
df_test['Engine'] = df_test['Engine'].fillna('1197 CC')


# In[16]:


plt.figure(figsize=(20,5))
data['Power'].value_counts().head(100).plot.bar()
plt.show()


# In[17]:


df_train['Power'] = df_train['Power'].fillna('74 bhp')
df_test['Power'] = df_test['Power'].fillna('74 bhp')

#Notice that above there are some null values also. 

df_train['Power'] = df_train['Power'].replace("null bhp", "74 bhp")
df_test['Power'] = df_test['Power'].replace("null bhp", "74 bhp")


# In[18]:


df_train.isnull().sum()


# In[19]:


df_test.isnull().sum()


# Now thath the null values are remoed we proceed to convert all the string columns to number:

# In[20]:


import re

def get_number(name):
    title_search = re.search('([\d+\.+\d]+\W)', name)
    
    if title_search:
        return title_search.group(1)
    return ""


# In[21]:


df_train['Mileage'] = df_train['Mileage'].apply(get_number).astype('float')
df_train['Engine'] = df_train['Engine'].apply(get_number).astype('int')
df_train['Power'] = df_train['Power'].apply(get_number).astype('float')

df_test['Mileage'] = df_test['Mileage'].apply(get_number).astype('float')
df_test['Engine'] = df_test['Engine'].apply(get_number).astype('int')
df_test['Power'] = df_test['Power'].apply(get_number).astype('float')


# In[28]:


from sklearn.model_selection import train_test_split

y = np.log1p(df_train.Price)  
X = df_train.drop(['Price'],axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.82,test_size=0.18,random_state=0)

X_train.head()


# In[23]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

X_train['Name'] = label_encoder.fit_transform(X_train['Name'])
X_valid['Name'] = label_encoder.transform(X_valid['Name'])
df_test['Name'] = label_encoder.fit_transform(df_test['Name'])

X_train['Location'] = label_encoder.fit_transform(X_train['Location'])
X_valid['Location'] = label_encoder.transform(X_valid['Location'])
df_test['Location'] = label_encoder.fit_transform(df_test['Location'])

X_train['Fuel_Type'] = label_encoder.fit_transform(X_train['Fuel_Type'])
X_valid['Fuel_Type'] = label_encoder.transform(X_valid['Fuel_Type'])
df_test['Fuel_Type'] = label_encoder.fit_transform(df_test['Fuel_Type'])

X_train['Transmission'] = label_encoder.fit_transform(X_train['Transmission'])
X_valid['Transmission'] = label_encoder.transform(X_valid['Transmission'])
df_test['Transmission'] = label_encoder.fit_transform(df_test['Transmission'])

X_train['Owner_Type'] = label_encoder.fit_transform(X_train['Owner_Type'])
X_valid['Owner_Type'] = label_encoder.transform(X_valid['Owner_Type'])
df_test['Owner_Type'] = label_encoder.fit_transform(df_test['Owner_Type'])


# In the above cell I don't know why it is giving the warning as I had to look to the Internet to do it manually.
# The label_encoder.fit_transform() , label_encoder.transform() were not working for some reason.
# My reasoning says its some anaconda warning to flag potentially confusing "chained" assignments.

# In[24]:


X_train.head()


# In[25]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

predictions = my_model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(predictions, y_valid)))
print("MSE: " + str(mean_squared_error(predictions, y_valid)))
print("MSLE: " + str(mean_squared_log_error(predictions, y_valid)))


# In[26]:


preds_test = my_model.predict(df_test)
preds_test = np.exp(preds_test)-
print(preds_test)

preds_test = preds_test.round(2)
print(preds_test)


# In[27]:


df_output = pd.DataFrame({'Price': preds_test})
df_output.to_excel('submission.xlsx', index=False)
df_output.head()


# Team Verzeo,
# 
# Thank You for taking the time to teach us the very basics of Data Science and Machine Learning.
# In the making of this particular project I had to visit google for various substitute ways for implementing the usual methods as they took a lot of time and also had to copy-paste few bits here and there (not going to lie , I never heard of "xgboost").
# 
# In the end I got to complete the project with knowing many ways to save time and how things actually work in the feild.
# Thank You again.
# 
# 
