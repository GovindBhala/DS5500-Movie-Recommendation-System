#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns


# # Loading Movie Lens datafiles

# In[2]:


movies=pd.read_csv("dataset2/movies.csv")
links =pd.read_csv("dataset2/links.csv")
ratings=pd.read_csv("dataset2/ratings.csv")


# #### DataFrame for filters

# In[3]:


tags=pd.read_csv("dataset2/tags.csv")
genome_scores =pd.read_csv("dataset2/genome-scores.csv")
genome_tags =pd.read_csv("dataset2/genome-tags.csv")


# ### Genome Tagging
# 
# 1,128 tags for 13,176 unique movies.
# 
# Tag genome records how strongly each tag applies to each movie on a continuous scale from 0 to 1.
# 
# 0 = does not applies at 1 = applies very strongly
# 

# In[4]:


genome = pd.merge(left=genome_scores, right=genome_tags, left_on='tagId', right_on='tagId')


# ##### Genome Tagging for Movie: Toy Story

# In[8]:


dataset = genome[genome['movieId']==1 & genome['tag'].isin(['toys','vampires','kids','cartoon','adventure','family','funny','violence','animation','james bond'])].sort_values(by ='relevance', ascending=False).head(10)
ax = plt.axes()
sns.barplot(y="tag", x="relevance", data=dataset)
ax.set(ylabel="Tag",
       xlabel="Relevance")
ax.set_title("Genome Tagging for Movie: Toy Story")
sns.despine(left=True, bottom=True)


# In[12]:


df = genome[genome['tag'].isin(['james bond'])].sort_values(by ='relevance', ascending=False).head(20)


# In[13]:


dataset = pd.merge(left=df, right=movies, left_on='movieId', right_on='movieId')


ax = plt.axes()
sns.barplot(y="title", x="relevance", data=dataset)
ax.set(ylabel="",
       xlabel="Relevance")
ax.set_title("James bond top 20 movies by genome relevance score")
sns.despine(left=True, bottom=True)


# In[10]:


genome.head()


# In[14]:


user_tags = st.multiselect('Show tags', df['tag'].unique())
# Filter dataframe
df = genome[genome['tag'].isin(user_tags)].sort_values(by ='relevance', ascending=False).head(20)
dataset = pd.merge(left=df, right=movies, left_on='movieId', right_on='movieId')

st.write(dataset)


# In[ ]:




