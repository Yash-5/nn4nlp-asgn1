#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import scipy.sparse as sp
import utils
from tqdm import tqdm
import sklearn
from sklearn import naive_bayes, linear_model
from collections import Counter


# In[2]:


data_dir = "./data/"
train_filename = "topicclass/topicclass_train.txt"
valid_filename = "topicclass/topicclass_valid.txt"
test_filename = "topicclass/topicclass_test.txt"


# In[3]:


train_X, train_y = utils.parse_file(data_dir + train_filename)
valid_X, valid_y = utils.parse_file(data_dir + valid_filename)
test_X = utils.parse_file(data_dir + test_filename, has_labels=False)


# In[4]:


vocab_size = len(utils.word2index) - 1


# In[5]:


sparse_train_X = sp.dok_matrix((len(train_X), vocab_size), dtype=np.int8)
sparse_valid_X = sp.dok_matrix((len(valid_X), vocab_size), dtype=np.int8)
sparse_test_X = sp.dok_matrix((len(test_X), vocab_size), dtype=np.int8)


# ## Baseline
# ### Naive Bayes with word counts and smoothing

# In[6]:


for i, w in tqdm(enumerate(train_X)):
    for j in w:
        sparse_train_X[i, j - 1] += 1
for i, w in tqdm(enumerate(valid_X)):
    for j in w:
        sparse_valid_X[i, j - 1] += 1
for i, w in tqdm(enumerate(test_X)):
    for j in w:
        sparse_test_X[i, j - 1] += 1


# In[7]:


index2label = {}
for x in utils.label2index.items():
    index2label[x[1]] = x[0]


# In[8]:


nbmodel = naive_bayes.MultinomialNB(alpha=0.25, fit_prior=False, class_prior=None)
nbmodel.fit(sparse_train_X, train_y)
nb_val_pred = nbmodel.predict(sparse_valid_X)
print(np.where(nb_val_pred == valid_y)[0].shape[0] / len(valid_y))


# In[9]:


nbmodel = naive_bayes.MultinomialNB(alpha=0.25, fit_prior=True, class_prior=None)
nbmodel.fit(sparse_train_X, train_y)
nb_val_pred = nbmodel.predict(sparse_valid_X)
print(np.where(nb_val_pred == valid_y)[0].shape[0] / len(valid_y))


# In[10]:


nb_test_pred = nbmodel.predict(sparse_test_X)


# In[11]:


with open('predict/nb/valid_preds', 'w') as f:
    for x in nb_val_pred:
        f.write(index2label[x] + '\n')
with open('predict/nb/test_preds', 'w') as f:
    for x in nb_test_pred:
        f.write(index2label[x] + '\n')


# In[ ]:




