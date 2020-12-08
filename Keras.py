#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import nltk
import praw
import json
import pprint
import pandas as pd
import numpy as np
import re
from io import StringIO
import matplotlib.pyplot as plt


# In[2]:


reddit = praw.Reddit(client_id='d82BgeGuxFNBlA', client_secret='Ddi18K33GxZoIJLXGPkqTRZiO3o',
                    password='Rohit295', user_agent='redrohit295',
                    username='redrohit295')


# In[3]:


list_of_items = []
fields = ['title','subreddit_id']

ritreddit = reddit.subreddit('rit')
for submission in ritreddit.top(limit=500):
    to_dict = vars(submission)
    sub_dict = {field: to_dict[field] for field in fields}
    list_of_items.append(sub_dict)


# In[4]:


with open('ritdata500keras.json', 'w') as f:
    json.dump(list_of_items, f)


# In[5]:


list_of_items = []
fields = ['title','subreddit_id']

ritreddit = reddit.subreddit('christmas')
for submission in ritreddit.top(limit=500):
    to_dict = vars(submission)
    sub_dict = {field: to_dict[field] for field in fields}
    list_of_items.append(sub_dict)


# In[6]:


with open('christmasdata500keras.json', 'w') as f:
    json.dump(list_of_items, f)


# In[7]:


ritfile = 'ritdata500keras.json'
with open(ritfile) as rit_file:
    rit_dict = json.load(rit_file)

# converting json dataset from dictionary to dataframe
rit_df = pd.DataFrame.from_dict(rit_dict)
rit_df.reset_index(level=0, inplace=True)


# In[8]:


chritmasfile = 'christmasdata500keras.json'
with open(chritmasfile) as chritmas_file:
    christmas_dict = json.load(chritmas_file)

# converting json dataset from dictionary to dataframe
christmas_df = pd.DataFrame.from_dict(christmas_dict)
christmas_df.reset_index(level=0, inplace=True)


# In[9]:


data_df = pd.concat([rit_df,christmas_df])


# In[10]:


data_df = data_df.replace('t5_2qh3x',0)
data_df = data_df.replace('t5_2qi2n',1)


# In[11]:


from sklearn.utils import shuffle
data_df = shuffle(data_df)


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_df["title"], data_df["subreddit_id"], test_size=0.5, random_state=1)

X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=1)


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train_k = vectorizer.transform(X_train)
X_dev_k  = vectorizer.transform(X_dev)


# In[14]:


from keras.models import Sequential
from keras import layers

input_dim = X_train_k.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[17]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[18]:


history = model.fit(X_train, y_train, epochs=100, verbose=False,validation_data=(X_test, y_test),batch_size=10)


# In[21]:


history = model.fit(X_train_k, y_train,epochs=100,verbose=False,validation_data=(X_dev_k, y_dev),batch_size=10)


# In[22]:


loss, accuracy = model.evaluate(X_train_k, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_dev_k, y_dev, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[ ]:




