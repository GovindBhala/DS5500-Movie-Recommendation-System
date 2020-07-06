#!/usr/bin/env python
# coding: utf-8

# # Item-Item Recommendations for Non-Users (Similar Movies)
# **Use Case**: User without a profile can get recommendations based on a movie they have previously enjoyed by displaying similar movies.    
# Details:
# - Allow users to put in a movie that they liked and output similar movies     
# - For a first pass, use a simple tf-idf genre-based recommendation model. Will update with movie profiles from personalized recommendations     
# - If same similarity score (aka same genres), recommend highest rated based on weighted average 
#     - Generally want diverse recommendations, expose long tail HOWEVER these people are already non-users who are not rating movies, thus want to give credibility to draw them in and get them to watch anything 
#     - Nothing to based the diverse recommendations on other than randomness. 
#     
# Process:
# - Create TF-IDF matrix based on genres
# - User enters free form text for movie title. Find 10 most similar title (at least 70% similar) and user chooses from dropdown 
# - Find 10 most similar movies. Secondarily sorted on weighted average. Display

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz


# # Cached Setup
# Called in main funciton     
# Set up TF-IDF matrix for genres and other needed data structures

# In[335]:


@st.cache(allow_output_mutation=True)
def cached_functions(df):

    # set up tf-idf and indices
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['genre_str'])
    # record list of movie Ids and indices
    movieIds = df['movieId']
    indices = pd.Series(df.index, index=df['movieId'])
    
    # get unique list of movies for user input 
    movies_unique = np.sort(df.title_year.unique())
    
    return movieIds, indices, tfidf_matrix, movies_unique


# ## Recommend movies based on cosine similarity of genre tf-idf
# - Input selected_id: movieId 
#     - User input: title (with year) from drop down menu. Need year because some titles are duplicate 
#     - movieId is unique, so good for the funciton input 
# - Find cosine similarity 
# - Merge with display data with metadata
# - Sort by scores first, weighted average second if same score

# In[355]:


def recommend_movies(df, selected_id, movieIds, indices, tfidf_matrix):
    
    # get index of movies in tfidf matrix
    idx = indices[selected_id]
    tfmat = tfidf_matrix[idx]
    
    # similarity between movie and other movies
    scores = list(linear_kernel(tfmat, tfidf_matrix)[0])

    # scores in order of movieIds, concat back together
    recs = pd.concat([movieIds, pd.Series(scores)], axis = 1, sort = False)
    recs.columns = ['movieId', 'score']

    # merge with overall data
    recs = pd.merge(recs, df, on = 'movieId')

    # remove original movie
    recs = recs[recs.movieId != selected_id]
    
    # sort by score first, then weighted average second
    # if same similarity score, recommend higher rated movie
    recs = recs.sort_values(['score', 'weighted_avg'], ascending = False)
    
    return recs


# # Streamlit App with User Input
# - User input: text input
# - Find options that are close to the text input based on fuzzy string matching
#     - Works for not-quite-right movie title and misspellings
#     - Top 10 IF similarity ratio > 70. Don't display anything if similarity too low
# - Select out of drop down. Drop down includes (year) 
#     - Some titles are duplicate so need year
# - Get movieId from selection and use that as recommendation input

# In[ ]:


def write(df, movieIds, indices, tfidf_matrix, movies_unique):

    st.title('Similar Movie Recommendations')
    st.header('View movies similar to movies that you have enjoyed in the past')
    st.write('Type a movie title hit Enter. You will see a list of potentially matching movies in our system.      \n' + 
             'Select your choice and select **Display Recommendations** to see similar movies.')
    
    # get user input text
    # too many movies for a full drop down 
    user_text = st.text_input("Movie Title")
    # downcase input
    user_text = user_text.lower()

    if user_text == '':
        st.write('Waiting for input')
    else:

        # fuzzy string matching to find similarity ratio between user input and actual movie title (downcased)
        # works for misspellings as well 
        # limit to 70% similarity 
        options = df.copy()
        options['sim'] = options.title_downcased.apply(lambda row: fuzz.token_sort_ratio(row, user_text))
        options = options[options.sim > 70].sort_values('sim', ascending = False).head(10).title_year.unique()

        # find movies that are similar to what they typed 
        if len(options) > 0:

            # select from dropdown 
            user_title = st.selectbox('Select Movie', options)

            if st.button('Display Recommendations'):
                # get recommendations
                recs = recommend_movies(df, df[df.title_year == user_title].movieId.values[0], movieIds, indices, tfidf_matrix)
                # top 10
                recs = recs.head(10)

                st.write(recs.drop(columns = ['movieId', 'weighted_avg', 'actors_downcased', 'directors_downcased',
                                              'title_downcased', 'title_year', 'score', 'genre_str', 'decade']))

        # if nothing > 70% similiarity, then can't find a matching movie
        else:
            st.write("Sorry, we can't find any matching movies")

