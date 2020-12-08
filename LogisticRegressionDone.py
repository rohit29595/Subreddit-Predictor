#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sklearn
import nltk
import praw
import json
import pprint
import pandas as pd
import numpy as np
import re
from io import StringIO     


# In[7]:


reddit = praw.Reddit(client_id='d82BgeGuxFNBlA', client_secret='Ddi18K33GxZoIJLXGPkqTRZiO3o',
                    password='Rohit295', user_agent='redrohit295',
                    username='redrohit295')


# In[8]:


list_of_items = []
fields = ['title','subreddit_id']

ritreddit = reddit.subreddit('rit')
for submission in ritreddit.top(limit=500):
    to_dict = vars(submission)
    sub_dict = {field: to_dict[field] for field in fields}
    list_of_items.append(sub_dict)


# In[9]:


with open('ritdata500lg.json', 'w') as f:
    json.dump(list_of_items, f)


# In[10]:


list_of_items = []
fields = ['title','subreddit_id']

ritreddit = reddit.subreddit('christmas')
for submission in ritreddit.top(limit=500):
    to_dict = vars(submission)
    sub_dict = {field: to_dict[field] for field in fields}
    list_of_items.append(sub_dict)


# In[11]:


with open('christmasdata500lg.json', 'w') as f:
    json.dump(list_of_items, f)


# In[12]:


ritfile = 'ritdata500lg.json'
with open(ritfile) as rit_file:
    rit_dict = json.load(rit_file)

# converting json dataset from dictionary to dataframe
rit_df = pd.DataFrame.from_dict(rit_dict)
rit_df.reset_index(level=0, inplace=True)


# In[13]:


chritmasfile = 'christmasdata500.json'
with open(chritmasfile) as chritmas_file:
    christmas_dict = json.load(chritmas_file)

# converting json dataset from dictionary to dataframe
christmas_df = pd.DataFrame.from_dict(christmas_dict)
christmas_df.reset_index(level=0, inplace=True)


# In[14]:


data_df = pd.concat([rit_df,christmas_df])


# In[15]:


data_df = data_df.replace('t5_2qh3x',0)
data_df = data_df.replace('t5_2qi2n',1)


# In[16]:


from sklearn.utils import shuffle
data_df = shuffle(data_df)


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_df["title"], data_df["subreddit_id"], test_size=0.5, random_state=1)

X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=1)


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train_lg = vectorizer.transform(X_train)
X_test_lg  = vectorizer.transform(X_test)


# In[19]:


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train_lg, y_train)
score = classifier.score(X_test_lg, y_test)
print("Accuracy:", score)

