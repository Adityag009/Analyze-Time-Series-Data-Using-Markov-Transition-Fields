#!/usr/bin/env python
# coding: utf-8

# #### The Electricity Transformer Temperature (ETT) dataset comprises the measurements of different parameters of transformers in China. The data was collected from July 2016 to July 2018. For this project, you will use the ETT-small dataset, which contains the data of two transformers. It consists of the following data columns:
# 
# Column	Description
# date	The date of the recorded sample
# 
# HUFL	High Useful Load
# 
# HULL	High Useless Load
# 
# MUFL	Medium Useful Load
# 
# MULL	Medium Useless Load
# 
# LUFL	Low Useful Load
# 
# LULL	Low Useless Load
# 
# OT	Oil Temperature

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pyts.preprocessing.discretizer import KBinsDiscretizer
from skimage import measure as sm
import warnings
warnings.simplefilter('ignore')
from matplotlib import cm ,colors


# In[6]:


df = pd.read_csv("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")


# In[8]:


df.head()


# In[18]:


df.info()


# In[9]:


len(df)


# In[10]:


df.shape


# In[12]:


ett = df.truncate(after = 999)


# In[13]:


fig = plt.figure(figsize=(28,4))
plt.plot(ett['date'],ett['HUFL'],linewidth=1)
plt.show()


# In[19]:


n_bins = 10
strategy = 'quantile'
discretizer = KBinsDiscretizer(n_bins = n_bins, strategy = strategy, raise_warning = False)
X = ett['HUFL'].values.reshape(1, -1)
ett['HUFL_disc'] = discretizer.fit_transform(X)[0]
# View the resulting dataframe
ett.head()


# In[23]:


ett['HUFL_disc'].unique()


# In[25]:


X.shape


# In[28]:


m_adj = np.zeros((n_bins,n_bins))
for k in range(len(ett.index)-1):
    #matrix Iteration
    index = ett['HUFL_disc'][k]
    next_index = ett['HUFL_disc'][k+1]
    m_adj[next_index][index] +=1
    
print(m_adj)


# In[31]:


ett.index


# In[32]:


index


# In[37]:


ett['HUFL_disc'][31]


# In[43]:


ett['HUFL_disc'][31]


# In[44]:


mtm = m_adj/m_adj.sum(axis=0)
print(mtm)


# In[45]:


n_t = len(ett.index)
mtf = np.zeros((n_t,n_t))

for i in range(n_t):
    for j in range(n_t):
        mtf[i,j]=mtm[ett['HUFL_disc'][i]][ett['HUFL_disc'][j]]*100


# In[46]:


mtf


# In[48]:


fig = plt.figure(figsize=(10,6))
plt.imshow(mtf)
plt.colorbar()
plt.show()


# In[49]:


mtf_reduced = sm.block_reduce(mtf,block_size=(10,10),func=np.mean)
fig = plt.figure(figsize=(8,8))
plt.imshow(mtf_reduced)
plt.colorbar()

plt.show()


# In[55]:


mtf_diag = [mtf_reduced[i][i] for i in range(len(mtf_reduced))]
fig, ax = plt.subplots(figsize = (28, 4))
norm = colors.Normalize(vmin=np.min(mtf_diag), vmax=np.max(mtf_diag))
cmap = cm.viridis
for i in range(0, n_t, 10):
    ax.plot(ett['date'][i:i+10+1], ett['HUFL_disc'][i:i+10+1], c = cmap(norm(mtf_diag[int(i/10)])))

# Optional
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax)    
plt.show()


# In[ ]:




