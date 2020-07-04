#!/usr/bin/env python
# coding: utf-8

# # Display Personalized Recommendations based on User ID and Filters

# In[2]:


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
from fuzzywuzzy import fuzz


# In[3]:


@st.cache(allow_output_mutation=True)
def load_data():
        
    df = pd.read_parquet('processed_df.parq')
    # version of ratings that has manually entered user profiles added on 
    ratings = pd.read_parquet('ratings_sample_useradd.parq')
    ratings = ratings.drop(columns = ['index', 'timestamp'])
    ratings = ratings.reset_index(drop = True)
        
    return df, ratings


# In[5]:


@st.cache(allow_output_mutation = True)
def create_ratings_df(new_ratings, new_users, new_movies, ratings):
            
    # create dataframe from lists of newly added from profile dadd
    d = {'rating':new_ratings, 'userId':new_users, 'movieId':new_movies}
    new_ratings = pd.DataFrame(d)
    
    # sometimes duplicate movies from user profile adds - average ratings. Else matrix multiplication won't work
    new_ratings = new_ratings.groupby(['userId', 'movieId']).rating.mean()   
    
    # concat with original ratings
    ratings = pd.concat([ratings, new_ratings])
    ratings = ratings.reset_index(drop = False)
    
    return ratings


# In[3]:


@st.cache(allow_output_mutation=True)
def user_content_recommendations(user_id, df, df_display, ratings):   
    
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
    movies_user = df[df.movieId.isin(ratings_user.movieId.unique())]
    watched = movies_user.movieId.unique()
    movies_user = movies_user.drop(columns = ['movieId', 'title_eng', 'year'])
    
    mean_rating = np.mean(ratings_user.rating)
    ratings_user.rating = ratings_user.rating - mean_rating

    profile = movies_user.T.dot(ratings_user.rating.values)
    
    recommendations = metrics.pairwise.cosine_similarity(df.drop(columns = ['movieId', 'title_eng', 'year']), 
                                                         np.asmatrix(profile.values))
    recommendations = pd.DataFrame(recommendations)
    recommendations.columns = ['prediction']
    recommendations = pd.merge(recommendations, df_display, 
                               left_index = True, right_index = True, how = 'left')
    #recommendations = recommendations.sort_values('prediction', ascending = False)
    recommendations = recommendations[~recommendations.movieId.isin(watched)]
    #recommen_ratings = pd.merge(recommendations,movies_raitings)
    
    # NEW ADDITION FOR APP: limit to recommendations similarity > 0 
        # don't recommend movies that are similar to movies they dislike
    recommendations = recommendations[recommendations.prediction > 0]
    
    return recommendations


# In[ ]:


def write(df_display, genres_unique, actors_df, directors_df, countries_unique,
          language_unique, tags_unique, new_ratings, new_users, new_movies, df, ratings):
    
    st.title('Personalized Movie Recommendations')
    st.write('Select **Display Recommendations** with no inputs to view your top recommendations. \n' + 
             'Or select filters to see your top recommended movies in those categories.')
    
    ratings = create_ratings_df(new_ratings, new_users, new_movies, ratings)
    
    userId = st.text_input('Enter your User ID:')
    
    
    if userId == '':
        st.write('Cannot provide recommendations without an ID')
    else:
        userId_int = int(userId)
            
        # check valid ID
        if userId_int not in set(ratings.userId.unique()) and userId_int not in set(new_users):
            st.write('Not a valid ID')
        else:
            # generate recommendations
            recommendation = user_content_recommendations(userId_int, df, df_display, ratings)
    
            ## filtering 
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
                df_filtered = recommendation[(recommendation.Genres.map(set(genre_input).issubset)) & 
                                         (recommendation['Filming Countries'].map(set(country_input).issubset)) &
                                         (recommendation['Language(s)'].map(set(language_input).issubset)) & 
                                         (recommendation.Tags.map(set(tag_input).issubset))  & 
                                         (recommendation['Actors'].map(set(actor_input).issubset)) &
                                         (recommendation['Director(s)'].map(set(director_input).issubset))
                                        ].sort_values('prediction', ascending = False).head(10).drop(columns = ['weighted_avg',
                                                                                                                'actors_downcased', 
                                                                                                                'directors_downcased',
                                                                                                                'title_downcased', 
                                                                                                                'title_year', 
                                                                                                                'movieId',
                                                                                                               'prediction',
                                                                                                                'genre_str'])
                # if no valid movies with combination of filters, notify. Else display dataframe
                if len(df_filtered) > 0:
                    st.write(df_filtered)
                else:
                    st.write('Found no recommended movies that match your selections')

