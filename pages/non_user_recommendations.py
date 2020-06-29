#!/usr/bin/env python
# coding: utf-8

# # Generate Recommendations for Non-Users & Streamlit

# 1. Top rated movies within a filter: weighted average of # of reviews and average ratings - DONE
#     - Genomes: only display if above X threshold? Or add relevance score into the weighted average?
# 
# UI elements to do:
# - Biographic information for the actor, director (IMDB) (?)
# 
# UI in the future: 
# - Filtering within personalized recommendations (user ID field)
# - Tab allowing users to input own ratings and get a recommendation out
# - EDA

# #### To Run:
# 1. Convert notebook to py file
#     - Run in command line: py -m jupyter nbconvert --to script streamlit_example.ipynb
# 2. Run streamlit app
#     - Run in command line: streamlit run streamlit_example.py

# In[1]:


import pandas as pd
import os
import numpy as np
import datetime as datetime
import operator
import streamlit as st
import fastparquet
import re
from fuzzywuzzy import fuzz


# ## Get Unique Lists of Filter Options

# In[2]:


def cat_list_expand(df, var):
    
    # expand lists such that one entry per row 
    expanded = df[[var, 'movieId']]
    expanded = pd.DataFrame({
        col:np.repeat(expanded[col].values, expanded[var].str.len()) for col in expanded.columns.drop(var)}
    ).assign(**{var:np.concatenate(expanded[var].values)})[expanded.columns]

    return expanded


# In[42]:


@st.cache(allow_output_mutation=True)
def unique_lists(df):
    
    # unique lists. Sort alphabetically
    genres_unique = np.sort(cat_list_expand(df, 'Genres').Genres.unique())
    countries_unique  = np.sort(cat_list_expand(df, 'Filming Countries')['Filming Countries'].unique())
    language_unique = np.sort(cat_list_expand(df, 'Language(s)')['Language(s)'].unique())
    tags_unique = np.sort(cat_list_expand(df, 'Tags').Tags.unique())
    
    # actors and directors: user input fuzzy string matching.
    # Get version with lower case for user matching + upper case for display
    actors_df = pd.merge(cat_list_expand(df, 'actors_downcased').actors_downcased,
                         cat_list_expand(df, 'Actors').Actors, left_index = True, right_index = True)
    # drop duplicated rows so unique
    actors_df = actors_df[actors_df.duplicated() == False]
    actors_df.columns = ['actors_downcased', 'actors_upcased']

    directors_df = pd.merge(cat_list_expand(df, 'directors_downcased').directors_downcased,
                            cat_list_expand(df, 'Director(s)')['Director(s)'], left_index = True, right_index = True)
    directors_df = directors_df[directors_df.duplicated() == False]    
    directors_df.columns = ['directors_downcased', 'directors_upcased']
    
    
    return genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique


# ## Display in Streamlit with filter options
# Display:
# - Title
# - Year
# - Description
# - Duration
# - Genres
# - Actors
# - Directors
# - Production Company
# - Country
# - Language
# - Genome Tags
# - Number of ratings
# - Average rating    
#    
# Filter by:
# - Genres
# - Actors
# - Directors
# - Country
# - Language
# - Genome Tags
# 
# Default table is highest rated movies without filters    
#    
#    
# Extensions:
# - AND/OR advanced search option? This might be difficult

# In[ ]:


def write(df_display, genres_unique, actors_df, directors_df, countries_unique,
          language_unique, tags_unique):
    
    st.title('Top Rated Movie Recommendations')
    st.header('Enter desired filters and select "Display Recommendations" \n')
    st.write('Please note filters use AND logic')
    st.write('If you wish to see overall top rated movies, select Display Recommendations without any filters')
    
    # get user inputs: multiple selection possible per category
    genre_input = st.multiselect('Select genre(s)', genres_unique)
    country_input = st.multiselect('Select filming country(s)', countries_unique)
    language_input = st.multiselect('Select language(s)', language_unique)
    tag_input = st.multiselect('Select genome tags(s)', tags_unique)

    # actors, directors get text inputs
    # Dropdowns too much for streamlit to handle
    # allow multiple entires
    actor_input = st.text_input('Type actor(s) names separated by comma. Select intended actor(s) from dropdown that appears')
    if actor_input != '':
        # downcase input
        actor_input = actor_input.lower()
        # split into list 
        actor_input = actor_input.split(', ')

        # fuzzy string matching to find similarity ratio between user input and actual actors (downcased)
        # works for misspellings as well 
        # limit to 70% similarity 
        options = []
        actors_sim = actors_df.copy()
        for i in actor_input:
            actors_sim['sim'] = actors_sim.actors_downcased.apply(lambda row: fuzz.ratio(row, i))
            options.append(actors_sim[actors_sim.sim > 70].sort_values('sim', ascending = False
                                                                      ).head(3).actors_upcased.unique())
        options = [item for sublist in options for item in sublist]    

        # list actors that are similar to what they typed
        if len(options) > 0:
            actor_input = st.multiselect('Select Actor(s)', options)
        else:
            st.write("Sorry, we can't find any matching actors")

    else:
        actor_input = []

    director_input = st.text_input('Type director(s) names separated by comma. Select intended director(s) from dropdown that appears')
    if director_input != '':
        # downcase input
        director_input = director_input.lower()
        # split into list 
        director_input = director_input.split(', ')

        # fuzzy string matching to find similarity ratio between user input and actual directors (downcased)
        # works for misspellings as well 
        # limit to 70% similarity 
        options = []
        directors_sim = directors_df.copy()
        for i in director_input:
            directors_sim['sim'] = directors_sim.directors_downcased.apply(lambda row: fuzz.ratio(row, i))
            options.append(directors_sim[directors_sim.sim > 70].sort_values('sim', ascending = False
                                                                            ).head(3).directors_upcased.unique())
        options = [item for sublist in options for item in sublist]    

        # list actors that are similar to what they typed
        if len(options) > 0:
            director_input = st.multiselect('Select Director(s)', options)
        else:
            st.write("Sorry, we can't find any matching directors")

    else:
        director_input = []

    # display recommendations once hit button
    if st.button('Display Recommendations'):
        # filter dataframe
        df_filtered = df_display[(df_display.Genres.map(set(genre_input).issubset)) & 
                                 (df_display['Filming Countries'].map(set(country_input).issubset)) &
                                 (df_display['Language(s)'].map(set(language_input).issubset)) & 
                                 (df_display.Tags.map(set(tag_input).issubset))  & 
                                 (df_display['Actors'].map(set(actor_input).issubset)) &
                                 (df_display['Director(s)'].map(set(director_input).issubset))
                                ].sort_values('weighted_avg', ascending = False).head(10).drop(columns = ['weighted_avg',
                                                                                                         'actors_downcased', 
                                                                                                          'directors_downcased',
                                                                                                         'title_downcased', 
                                                                                                         'title_year', 
                                                                                                          'movieId',
                                                                                                         'genre_str'])
        # if no valid movies with combination of filters, notify. Else display dataframe
        if len(df_filtered) > 0:
            st.write(df_filtered)
        else:
            st.write('Found no movies that match your selections')

