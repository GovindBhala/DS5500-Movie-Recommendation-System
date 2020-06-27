#!/usr/bin/env python
# coding: utf-8

# # Main Streamlit HomePage
# 5 pages:
# - Top rated movies with filtering 
# - Item-Item recommendations
# - Personalized recommendations
# - User profile creation: provide ratings
# - Visualizations

# In[5]:


import streamlit as st 
import pandas as pd
import os
import numpy as np
import operator
import fastparquet
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ### Import Individual Pages

# In[ ]:


import pages.home_page
import pages.non_user_recommendations
import pages.item_item_rec_app


# ## Run cached set up functions

# In[ ]:


df_toprated, genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique = pages.non_user_recommendations.cached_functions()


# In[ ]:


df_itemitem_orig, movieIds, indices, tfidf_matrix, movies_unique = pages.item_item_rec_app.cached_functions()
# copy so that doesn't cause cacheing error 
df_itemitem = df_itemitem_orig.copy()


# # Main Function: Navigation between Pages
# - Side radio button. Upon selection, call write() function within each page. Pass in arguments from cached calls. 

# In[ ]:


PAGES = ['Home', 'Top Rated Movies', 'Movie Based Recommendations']


# In[12]:


def main():
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", PAGES)    
    
    if selection == 'Home':
        pages.home_page.write()
    if selection == 'Top Rated Movies':
        pages.non_user_recommendations.write(df_toprated, genres_unique, actors_df, 
                                             directors_df, countries_unique, language_unique, tags_unique)
    if selection == 'Movie Based Recommendations':
        pages.item_item_rec_app.write(df_itemitem, movieIds, indices, tfidf_matrix, movies_unique)


# In[ ]:


main()

