#!/usr/bin/env python
# coding: utf-8

# #  Pre modeling : Data Preprocessing and Feature Exploration in python

# A) TASK# given attributes about the person to say their income is <=50k or >50k

# In[1]:


import pandas as pd
import numpy as np 


# r is used before the address to declare it is a raw string where backslashes are not interpret the address

# In[2]:


df= pd.read_csv(r"C:\Users\User\Downloads\adult.csv\adult.csv")


# In[3]:


df.head()


# In[4]:


print(df["income"].value_counts())


# In[5]:


# if income <=50k then it will show as 0 or it will >50k it shows as 1
df['income'] = [0 if x=='<=50k' else 1 for x in df['income']]
    


# In[6]:


# we are creating x as the data frame of features and  Y as the series
x = df.drop('income',1)
y = df.income


# In[7]:


print(x.head())


# In[8]:


print(y.head(10))


# # Data cleaning
# **B) the next process is to clean by removing duplicates, feature scaling, data types and so on

# In[9]:


print(x['education'].head())


# In[10]:


# one of the important technique  in python is label encoding/ one hot encoding/ get dummies
# it can change categorical features into 0's and 1's 

print(pd.get_dummies(x['education']).head())


# In[11]:


#  next process to decide which category we are going to use to predict our model
# first have to see unique features and predict which suits the most



for col_name in x.columns:
    if x[col_name].dtypes =='object':
        unique_cat = len(x[col_name].unique())
        print ("Feature {col_name} has {unique_cat} unique categories". format (col_name=col_name, unique_cat =unique_cat))
        

# here we assigning x columns as col_name and checking for the unique categories and count its length
# after those xcolumns and unique categories will be printed in format like . format (col_name=col_name, unique_cat =unique_cat)
# and using placeholders the return statement was printed
# In[12]:


# we found that native-country has 42 unique categories which is a high frequency than the others.
# so now check the native country column
print(x['native-country'].value_counts().sort_values(ascending=False).head(10))


# In[13]:


#if there is many low  frequency categories then those things are categorized into one single thing as dummy
#so the maximum or high probability frequency will help to predict the model \
# so the remaining low frequency are categorised as other
x['native-country'] = ['United-States' if x== 'United-States'else 'Other' for x in x['native-country']]
print(x['native-country'].value_counts().sort_values(ascending=False))    


# In[14]:


# now we create native country feature to use further so other features have to be dummmy. 
todummy_list = ['workclass','fnlwgt','education','educational-num','marital-status','occupation','relationship',
                'race','gender','capital-gain','capital-loss','hours-per-week','native-country','income']


# In[15]:


def dummy_df(df,todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x],prefix=x,dummy_na= False)
# Remove the original categorical column from the DataFrame
        df = df.drop(x,1)        
# Concatenate the one-hot encoded columns to the DataFrame
        df = pd.concat([df,dummies], axis=1)
        return df


# In[16]:


x= dummy_df(x, todummy_list)
print(x.head())


# # Removing null values
# 

# In[17]:


x.isnull().sum().sort_values(ascending = False).head(25)


# In[18]:


# so there is no null values in our dataset

df.info()


# In[ ]:




