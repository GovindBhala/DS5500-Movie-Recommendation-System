#!/usr/bin/env python
# coding: utf-8

# # Main Streamlit HomePage
# 5 pages:
# - Top rated movies with filtering 
# - Item-Item recommendations
# - Personalized recommendations
# - User profile creation: provide ratings
# - Visualizations   

# In[1]:


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
import pages.personalized_rec_app
import pages.profile_add_app
import pages.EDA_Streamlit_page


# ## Set up data and unique lists for filtering 
# Needed for multiple of the pages, so more efficient to only load once

# In[3]:


@st.cache(allow_output_mutation=True)
def data_setup():
    # read in data created in recommendation_data_display.ipynb
    df = pd.read_parquet('recommendation_display.parq')
    
    # recombine genre lists to string for tf-idf
    df['genre_str'] = df.Genres.apply(lambda row: ' '.join(row))

    # get unique lists of all filter values
    genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique = pages.non_user_recommendations.unique_lists(df)
    
    # data for item-item recommendations
    movieIds, indices, tfidf_matrix, movies_unique = pages.item_item_rec_app.cached_functions(df)
    
    # data for personalized recommendations
    df_dummies, ratings_df = pages.personalized_rec_app.load_data()
    
    return df, genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique, movieIds, indices, tfidf_matrix, movies_unique, df_dummies, ratings_df


# In[1]:


df, genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique, movieIds, indices, tfidf_matrix, movies_unique, df_dummies, ratings_df = data_setup()


# ## Set up Empty New User Profile

# In[ ]:


# function creates empty lists, but allow output mutation such that not overwritten when page refreshes
# only works with mutable data types
@st.cache(allow_output_mutation=True)
def list_create():
    return [], [], [], [], []


# In[ ]:


@st.cache(allow_output_mutation=True)
def empty_profile_create(ratings_df):

    # empty lists to hold user input: will persist across user refresh because of function
    new_ratings, new_users, new_movies, new_titles, userId_new = list_create()    

    # generate a new user id 
    # append to list because changes every time the page is run. Only want first max entry. 
    userId_new.append(int(ratings_df.userId.max() + 1))
    
    return new_ratings, new_users, new_movies, new_titles, userId_new


# In[ ]:


new_ratings, new_users, new_movies, new_titles, userId_new = empty_profile_create(ratings_df)


# # Main Function: Navigation between Pages
# - Side radio button. Upon selection, call write() function within each page. Pass in arguments from cached calls. 

# In[ ]:


PAGES = ['Home', 'Top Movie Visualizations', 'Top Rated Movies', 'Movie Based Recommendations',
         'Personalized Recommendations', 'Add Profile']


# In[12]:


def main(df, genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique,
         movieIds, indices, tfidf_matrix, movies_unique, df_dummies, ratings_df,
         new_ratings, new_users, new_movies, new_titles, userId_new):
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", PAGES)    
    
    if selection == 'Home':
        pages.home_page.write()
    if selection == 'Top Movie Visualizations':
        pages.EDA_Streamlit_page.write()
    if selection == 'Top Rated Movies':
        pages.non_user_recommendations.write(df, genres_unique, actors_df, 
                                             directors_df, countries_unique, language_unique, tags_unique)
    if selection == 'Movie Based Recommendations':
        pages.item_item_rec_app.write(df, movieIds, indices, tfidf_matrix, movies_unique)
    if selection == 'Personalized Recommendations':
        pages.personalized_rec_app.write(df, genres_unique, actors_df, directors_df, countries_unique,
                                         language_unique, tags_unique, new_ratings, new_users, new_movies, df_dummies,
                                         ratings_df)
    if selection == 'Add Profile':
        new_ratings, new_users, new_movies, new_titles = pages.profile_add_app.write(df, new_ratings, new_users,
                                                                                     new_movies, new_titles, userId_new)
        
    return new_ratings, new_users, new_movies, new_titles


# In[ ]:


new_ratings, new_users, new_movies, new_titles = main(df, genres_unique, actors_df, directors_df, countries_unique,
                                                      language_unique, tags_unique,
                                                      movieIds, indices, tfidf_matrix, movies_unique, df_dummies, ratings_df,
                                                      new_ratings, new_users, new_movies, new_titles, userId_new)

