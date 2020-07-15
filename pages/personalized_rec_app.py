#!/usr/bin/env python
# coding: utf-8

# # Generate & Display Personalized Recommendations
# **Use Case**: User with existing profile OR generated one on Add Profile page
# - Input: User ID 
# - Generate recommendations based on content-system: cosine similarity between user and movie profile. 
# - Allow filtering of recommendations based on desired movie attributes. 
#     - Do not display recommendations with cosine similarity < 0 
# 
# *Challenges*: very slow process to generate recommendations because of matrix multiplication.  
# Cannot pre generate similarity profiles because already running into memory limitations.   
#     
# Process:
# - Combine ratings data with any newly created profiles
# - User enters ID
# - Check if valid ID 
# - Generate recommendations
# - Allow user to filter down recommendations 
# - Display recommendations 

# In[1]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import operator
import scipy.spatial.distance as distance
from sklearn import metrics 
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import fastparquet
import streamlit as st
import pickle
import scipy
from fuzzywuzzy import fuzz
import sklearn


# ## Load Data (cached)
# Called in main app 

# In[4]:


@st.cache(allow_output_mutation=True)
def load_data():
      
    # sparse movie dataframe with attached metadata (column titles, movieIds in row order)
    df = scipy.sparse.load_npz("processed_df_sparse.npz")
    with open('sparse_metadata', "rb") as f:
        cols = pickle.load(f)
        movieIds = pickle.load(f)
    # version of ratings that has manually entered user profiles added on 
    ratings = pd.read_parquet('ratings_sample_useradd.parq')
    ratings = ratings.reset_index(drop = True)
        
    return df, ratings, cols, movieIds


# ## Combine ratings data with new profile created 

# In[6]:


@st.cache(allow_output_mutation = True)
def create_ratings_df(new_ratings, new_users, new_movies, ratings):
            
    # create dataframe from lists of newly added from profile add
    d = {'rating':new_ratings, 'userId':new_users, 'movieId':new_movies}
    new_ratings = pd.DataFrame(d)
    
    # sometimes duplicate movies from user profile adds -> average ratings. Else matrix multiplication won't work
    new_ratings = new_ratings.groupby(['userId', 'movieId']).rating.mean()  
    new_ratings = new_ratings.reset_index(drop = False)
    
    # concat with original ratings
    ratings = pd.concat([ratings, new_ratings], sort = False)
    ratings = ratings.reset_index(drop = True)

    return ratings


# ## Generate Recommendations 
# - Limit to specified user
# - Normalize ratings for specific user. If rating < mean, normalized ratings will be negative 
# - Create use profile: sum of ratings for each attribute
# - Generate recommendations: cosine similarity between movie and user profile 
# - Remove movies already watched/rated
# - Limit recommendations to similarity > 0 so that when filtering, don't display something they would DISlike 

# In[14]:


@st.cache(allow_output_mutation=True)

def user_content_recommendations(user_id, df, df_display, ratings, movieIds):   
    """
    ratings_user: limit to one user
    
    movies_user: movies rated by that user
    
    watched: keep track of movies already watched
    
    normalize ratings: subtract mean rating  from ratings
                       if rating < mean, normalized rating will be negative. Which is worse than 0 aka not rating movie at all.
    
    profile:create user profile: multiply item profile by user ratings --> sum of ratings for each attribute 
    
    recommendations: cosine similarity between movie and user profile 
                     merge to get title
                     sort
                     remove recommendations already watched
    """
    ratings_user = ratings[ratings.userId == user_id]
    ratings_user = ratings_user.sort_values('movieId')
    watched = ratings_user.movieId.unique()
    watched_index = [movieIds.index(i) for i in watched]
    movies_user = df[watched_index, :]
        
    mean_rating = np.mean(ratings_user.rating)
    ratings_user.rating = ratings_user.rating - mean_rating
    
    profile = scipy.sparse.csr_matrix(movies_user.T.dot(ratings_user.rating.values))
    
    # normalize profile to account for different numbers of ratings
    profile = sklearn.preprocessing.normalize(profile, axis = 1, norm = 'l2')
    
    # find similarity between profile and movies 
    # cosine similarity except movies not normalized 
    recommendations = df.dot(profile.T).todense()
    
    recommendations = pd.DataFrame(recommendations)
    recommendations = pd.merge(recommendations, pd.Series(movieIds).to_frame(), left_index = True, right_index = True)
    recommendations.columns = ['prediction', 'movieId']
    recommendations = recommendations[~recommendations.movieId.isin(watched)]
    recommendations = pd.merge(recommendations, df_display, on = 'movieId', how = 'left')
    #recommendations = recommendations.sort_values('prediction', ascending = False)
    #recommen_ratings = pd.merge(recommendations,movies_raitings, left_on = 'movieId', right_on = 'id')

    # NEW ADDITION FOR APP: limit to recommendations similarity > 0 
        # don't recommend movies that are similar to movies they dislike
    recommendations = recommendations[recommendations.prediction > 0]

    
    return recommendations


# ## Streamlit App
# 

# In[ ]:


def write(df_display, genres_unique, actors_df, directors_df, countries_unique,
          language_unique, tags_unique, decades_unique, new_ratings, new_users, new_movies, df, ratings, movieIds):
    
    # user instructions 
    st.title('Personalized Movie Recommendations')
    st.write('Select **Display Recommendations** with no inputs to view your top recommendations. \n' + 
             'Or select filters to see your top recommended movies in those categories.')
    
    # combine original ratings with newly created profiles
    ratings = create_ratings_df(new_ratings, new_users, new_movies, ratings)

    # user enter their user ID
    userId = st.text_input('Enter your User ID:')
    
    if userId == '':
        st.write('Cannot provide recommendations without an ID')
    else:
        # check if valid integer. If yes, convert
        try:
            userId_int = int(userId)
        # if cannot convert to an integer 
        except ValueError:
            st.write('Not a valid ID')
            
        # if valid integer, check if valid ID
        else: 
            
            # check valid ID
            if userId_int not in set(ratings.userId.unique()):
                st.write('Not a valid ID')
                
            # if valid ID, give recommendations 
            else:
                # generate recommendations
                recommendation = user_content_recommendations(userId_int, df, df_display, ratings, movieIds)

                ## filtering 
                # get user inputs: multiple selection possible per category except decade
                genre_input = st.multiselect('Select genre(s)', genres_unique)
                decade_input = st.selectbox('Select film decade', ['Choose an option'] + list(decades_unique))
                country_input = st.multiselect('Select filming country(s)', countries_unique)
                language_input = st.multiselect('Select language(s)', language_unique)
                tag_input = st.multiselect('Select genome tags(s)', tags_unique)

                # actors, directors get text inputs
                # Dropdowns too much for streamlit to handle
                # allow multiple entires
                actor_input = st.text_input('Type actor(s) names separated by comma. ' + 
                                            'Select intended actor(s) from dropdown that appears')
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
                        actors_sim['sim'] = actors_sim.actors_downcased.apply(lambda row: fuzz.token_sort_ratio(row, i))
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

                director_input = st.text_input('Type director(s) names separated by comma. ' + 
                                               'Select intended director(s) from dropdown that appears')
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
                        directors_sim['sim'] = directors_sim.directors_downcased.apply(lambda row: fuzz.token_sort_ratio(row, i))
                        options.append(directors_sim[directors_sim.sim > 70].sort_values('sim', ascending = False
                                                                                        ).head(3).directors_upcased.unique())
                    options = [item for sublist in options for item in sublist]    

                    # list directors that are similar to what they typed
                    if len(options) > 0:
                        director_input = st.multiselect('Select Director(s)', options)
                    else:
                        st.write("Sorry, we can't find any matching directors")

                else:
                    director_input = []

                # display recommendations once hit button
                if st.button('Display Recommendations'):
                    
                    # for decade, only filter if chose an option (no NA default for selectbox)
                    if decade_input != 'Choose an option':
                        rec_filtered = recommendation[(recommendation.decade ==  decade_input)]
                    else:
                        rec_filtered = recommendation.copy()
                    # filter dataframe
                    rec_filtered = rec_filtered[(rec_filtered.Genres.map(set(genre_input).issubset)) & 
                                                (rec_filtered['Filming Countries'].map(set(country_input).issubset)) &
                                                (rec_filtered['Language(s)'].map(set(language_input).issubset)) & 
                                                (rec_filtered.Tags.map(set(tag_input).issubset))  & 
                                                (rec_filtered['Actors'].map(set(actor_input).issubset)) &
                                                (rec_filtered['Director(s)'].map(set(director_input).issubset)) 
                                               ].sort_values('prediction', ascending = False
                                                            ).head(10).drop(columns = ['weighted_avg', 'actors_downcased', 
                                                                                       'directors_downcased', 'title_downcased', 
                                                                                       'title_year', 'movieId', 'prediction',
                                                                                       'genre_str', 'decade'])
                    # if no valid movies with combination of filters, notify. Else display dataframe
                    if len(rec_filtered) > 0:
                        st.write(rec_filtered)
                    else:
                        st.write('Found no recommended movies that match your selections')

