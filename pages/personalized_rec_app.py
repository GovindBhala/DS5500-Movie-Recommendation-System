#!/usr/bin/env python
# coding: utf-8

# # Generate & Display Personalized Recommendations
# **Use Case**: User with existing profile OR generated one on Add Profile page. Provide personalized recommendations using models defined in modules. 
# 
# Process:
# - Combine ratings data with any newly created profiles
# - User enters ID
# - Check if valid ID (in ratings dataset)
# - Generate recommendations
#     - Do not display recommenations with cosine similarity < 0 even if fit filter
# - Allow user to filter down recommendations 
# - Display recommendations    
#    
# Note: if run this locally outside of app, data paths will be incorrect including recommendation system modules. Assuming running in streamlit, in which case main_app.py calls these scripts from the root folder, which is where the datasets live.    
# Also, data is being passed in from main_app, so not all required data is loaded/created in this script

# In[20]:


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


# In[ ]:


# import recommendation system (py scripts)
from content_based_recommendations import user_content_recommendations


# ## Load Data
# Called in data prep section main app     
# - Load two sparse matrices for the combined model. df1 is movie profiles with one hot encoded genre, (top 3) actors, directors. df2 is movie profiles with one hot encoded top 5 tfidf tokens from description+genome tags
# - Load corresponding columns and movieIds (row) for sparse matrices. Don't need columns and movieIds are identical in the two datasets, so load just for df1
#     - All movie ids in both datasets because want to generate user profile based on all movies they have rated. Then filter recommendations down to the target group
# - Load in ratings data. Version with user profiles added on from prior runs of app
# - Load lists of movieIds with and without tags. Will generate recommendations from tagged movies with df2 and untagged movies with df1

# In[22]:


@st.cache(allow_output_mutation=True)
def load_data():
      
    # sparse movie dataframe with attached metadata (column titles, movieIds in row order)
    # two datasets for combined models 
    df1 = scipy.sparse.load_npz("processed_df_sparse.npz")
    df2 = scipy.sparse.load_npz("processed_df_text_sparse.npz")
    
    with open('sparse_metadata', "rb") as f:
        cols1 = pickle.load(f)
        movieIds = pickle.load(f)

    # preloaded collaborative filtering predictions
    collab_predictions = pd.read_parquet('Predictions/KNN_predictions_df.parq')
    collab_predictions = collab_predictions.rename(columns = {'est':'prediction', 'uid':'userId', 'iid':'movieId'})
    collab_predictions = collab_predictions.drop(columns = ['r_ui', 'details.actual_k', 'details.was_impossible'])
        
    # version of ratings that has manually entered user profiles added on 
    ratings = pd.read_parquet('ratings_sample_useradd.parq')
    ratings = ratings.reset_index(drop = True)
    
    # load movieId lists for movies with and without tags so can specify which movies to keep for which models
    with open('movieIds_tags', "rb") as f:
        movieIds_tags = pickle.load(f)
    with open('movieIds_notags', "rb") as f:
        movieIds_notags = pickle.load(f)
        
    return ratings, movieIds, df1, df2, collab_predictions, movieIds_tags, movieIds_notags


# ## Combine ratings data with new profile created in this run of the app
# - If user entered any new ratings in Profile Add tab, they will be in lists of the ratings, userId, and movieIds
# - Create a dataframe and append onto existing ratings data to use for recommendation generation

# In[68]:


@st.cache(allow_output_mutation = True)
def create_ratings_df(new_ratings, new_users, new_movies, ratings):
                
    # create dataframe from lists of newly added from profile add
    d = {'rating':new_ratings, 'userId':new_users, 'movieId':new_movies}
    new_ratings = pd.DataFrame(d)
    
    # sometimes duplicate movies from user profile adds if they enter hte same movie twice
        # take average of duplicate ratings. Else matrix multiplication won't work
    new_ratings = new_ratings.groupby(['userId', 'movieId']).rating.mean()  
    new_ratings = new_ratings.reset_index(drop = False)
    
    # concat with original ratings
    ratings = pd.concat([ratings, new_ratings], sort = False)
    ratings = ratings.reset_index(drop = True)

    return ratings


# ## Generate Recommendations 
# - Generate recommendations from specified system 
# - Limit recommendations to similarity > 0 so that when filtering, don't display something they would DISlike 
# - Merge with df_display to get features that we will disply on UI 

# In[25]:


@st.cache(allow_output_mutation = True)
def content_recommendations(user_id, df1, df2, df_display, ratings, movieIds, keep_movies1, keep_movies2): 
    
    recommend1 = user_content_recommendations(user_id, df1, ratings, movieIds, df_display, keep_movies1)
    recommend2 = user_content_recommendations(user_id, df2, ratings, movieIds, df_display, keep_movies2)
    
    # limit to recommendations similarity > 0 
        # don't recommend movies that are similar to movies they dislike
    recommend1 = recommend1[recommend1.prediction > 0]
    recommend2 = recommend2[recommend2.prediction > 0]

    # merge with display features
    #recommend1 = pd.merge(recommend1, df_display, on = 'movieId', how = 'left')
    #recommend2 = pd.merge(recommend2, df_display, on = 'movieId', how = 'left')

    return recommend1, recommend2


# In[63]:


@st.cache(allow_output_mutation = True)
def collab_content_recommendations(user_id, df1, collab_predictions, df_display, ratings, movieIds): 
    
    collab_rec = collab_predictions[collab_predictions.userId == user_id]

    # find movies in full set that are not in collaborative filtering predictions for this user
    keep_movies = set(movieIds).difference(set(collab_rec.movieId.unique()))
    
    # generate recommendations from content model with movies not in collab filtering
    content_rec = user_content_recommendations(user_id, df1, ratings, movieIds, df_display, keep_movies)
    
    # limit content recs to similarity > 0 
    content_rec = content_rec[content_rec.prediction > 0]

    # limit collabs recs to predicted rating > user's average rating (same as 0 marker above)
    collab_rec = collab_rec[collab_rec.prediction > ratings[ratings.userId == user_id].rating.mean()]
    
    # merge with display features
    collab_rec = pd.merge(collab_rec, df_display, on = 'movieId', how = 'left')
    #content_rec = pd.merge(content_rec, df_display, on = 'movieId', how = 'left')
    
    return collab_rec, content_rec


# ## Streamlit App
# - See notes on filtering options in non_user_recommendations script/notebook. Identical filter options here. 
# - Combine existing ratings with profiles newly created in the 'add profile' tab of UI. Then if enter userId generated there, will be able to produce recommendations

# In[5]:


def write(df_display, genres_unique, actors_df, directors_df, countries_unique,
          language_unique, tags_unique, decades_unique, new_ratings, new_users, new_movies, ratings, movieIds,
          collab_predictions, df1, df2, keep_movies1, keep_movies2):
    
    # user instructions 
    st.title('Personalized Movie Recommendations')
    st.write('Select **Display Recommendations** with no inputs to view your top recommendations. \n' + 
             'Or select filters to see your top recommended movies in those categories.')
    
    # combine original ratings with newly created profiles
    ratings = create_ratings_df(new_ratings, new_users, new_movies, ratings)

    # user enter their user ID
    userId = st.text_input('Enter your User ID:')
    
    # initial state is ''
    if userId == '':
        st.write('Cannot provide recommendations without an ID')
    else:
        # check if valid integer. If yes, convert
        try:
            userId_int = int(userId)
        # if cannot convert to an integer 
        except ValueError:
            st.write('Not a valid ID')
            
        # if valid integer, find if ID is in collaborative filtering set or not. Newly entered users will be in collab filter.
        else: 
            #set(collab_predictions.userId.unique())
            if userId_int in set(ratings.userId.unique()).intersection(set(collab_predictions.userId.unique())):
                
                # generate recommendations form collab-content combined model
                recommend1, recommend2 = collab_content_recommendations(userId_int, df1, collab_predictions, 
                                                                        df_display, ratings, movieIds)
                recommend1 = recommend1.drop(columns = ['userId'])
                
            elif userId_int in set(ratings.userId.unique()):
                
                # generate recommendations from combined content model 
                recommend1, recommend2 = content_recommendations(userId_int, df1, df2, df_display, ratings, movieIds,
                                                                 keep_movies1, keep_movies2)

            else:
                st.write('Not a valid ID')
                recommend1 = pd.DataFrame()
                
            if len(recommend1) > 0: 
            
                ## filtering 
                # get user inputs: multiple selection possible per category except decade
                # input sorted list of unique options 
                genre_input = st.multiselect('Select genre(s)', genres_unique)
                decade_input = st.selectbox('Select film decade', ['Choose an option'] + list(decades_unique))
                country_input = st.multiselect('Select filming country(s)', countries_unique)
                language_input = st.multiselect('Select language(s)', language_unique)
                tag_input = st.multiselect('Select genome tags(s)', tags_unique)

                # actors, directors get text inputs - dropdowns too many values for streamlit to handle
                # allow multiple entries with a commoa 
                actor_input = st.text_input('Type actor(s) names separated by commas. ' +
                                            'Select intended actor(s) from dropdown that appears')
                if actor_input != '':

                    # downcase input
                    actor_input = actor_input.lower()
                    # split into list based on commas
                    actor_input = actor_input.split(', ')

                    # fuzzy string matching to find similarity ratio between user input and actual actors (downcased)
                        # works for misspellings as well 
                        # limit to 70% similarity 
                    options = []
                    actors_sim = actors_df.copy()
                    for i in actor_input:
                        # find similarity ratio between input and all unique actors (downcased)
                        actors_sim['sim'] = actors_sim.actors_downcased.apply(lambda row: fuzz.token_sort_ratio(row, i))
                        # get top 3 with similarity > 70%
                        options.append(actors_sim[actors_sim.sim > 70].sort_values('sim', ascending = False
                                                                                  ).head(3).actors_upcased.unique())
                    # flatten options list
                    options = [item for sublist in options for item in sublist]    

                    # list actors that are similar to what they typed and accept user selection(s)
                    if len(options) > 0:
                        actor_input = st.multiselect('Select Actor(s)', options)
                    else:
                        st.write("Sorry, we can't find any matching actors")

                else:
                    actor_input = []

                director_input = st.text_input('Type director(s) names separated by commas. ' + 
                                               'Select intended director(s) from dropdown that appears')
                if director_input != '':
                    # downcase input
                    director_input = director_input.lower()
                    # split into list 
                    director_input = director_input.split(', ')

                    # fuzzy string matching to find similarity ratio between user input and actual actors (downcased)
                        # works for misspellings as well 
                        # limit to 70% similarity 
                    options = []
                    directors_sim = directors_df.copy()
                    for i in director_input:
                        # find similarity ratio between input and all unique directors (downcased)
                        directors_sim['sim'] = directors_sim.directors_downcased.apply(
                            lambda row: fuzz.token_sort_ratio(row, i))
                        # get top 3 with similarity > 70%
                        options.append(directors_sim[directors_sim.sim > 70].sort_values('sim', ascending = False
                                                                                        ).head(3).directors_upcased.unique())
                    # flatten options list
                    options = [item for sublist in options for item in sublist]    

                    # list directors that are similar to what they typed and accept user selection(s)
                    if len(options) > 0:
                        director_input = st.multiselect('Select Director(s)', options)
                    else:
                        st.write("Sorry, we can't find any matching directors")

                else:
                    director_input = []

                # display recommendations once hit button
                if st.button('Display Recommendations'):
                
                    # filter dataframe with rest of filters, sort and get top 10. Drop columns we don't want to display.
                    rec1_filtered = recommend1[(recommend1.Genres.map(set(genre_input).issubset)) & 
                                                (recommend1['Filming Countries'].map(set(country_input).issubset)) &
                                                (recommend1['Language(s)'].map(set(language_input).issubset)) & 
                                                (recommend1.Tags.map(set(tag_input).issubset))  & 
                                                (recommend1['Actors'].map(set(actor_input).issubset)) &
                                                (recommend1['Director(s)'].map(set(director_input).issubset)) 
                                               ].sort_values(['prediction', 'weighted_avg'], ascending = False)
                    rec2_filtered = recommend2[(recommend2.Genres.map(set(genre_input).issubset)) & 
                                                (recommend2['Filming Countries'].map(set(country_input).issubset)) &
                                                (recommend2['Language(s)'].map(set(language_input).issubset)) & 
                                                (recommend2.Tags.map(set(tag_input).issubset))  & 
                                                (recommend2['Actors'].map(set(actor_input).issubset)) &
                                                (recommend2['Director(s)'].map(set(director_input).issubset)) 
                                               ].sort_values(['prediction', 'weighted_avg'], ascending = False)
                    # for decade, only filter if chose an option (no NA default for selectbox)
                    if decade_input != 'Choose an option':
                        rec1_filtered = rec1_filtered[(rec1_filtered.decade == decade_input)]
                        rec2_filtered = rec2_filtered[(rec2_filtered.decade == decade_input)]
                        
                    # combine: 5 from each
                    if len(rec1_filtered) >= 5 and len(rec2_filtered) >= 5:
                        rec_filtered = pd.concat([rec1_filtered.head(int(5)), rec2_filtered.head(int(5))])  
                    elif len(rec1_filtered) < 5:
                        rec_filtered = pd.concat([rec1_filtered, rec2_filtered.head(int(10 - len(rec1_filtered)))])  
                    elif len(rec2_filtered) < 5:
                        rec_filtered = pd.concat([rec1_filtered.head(int(10 - len(rec1_filtered))), rec2_filtered])  

                    # sort based on weighted average
                    rec_filtered = rec_filtered.sort_values('weighted_avg', ascending = False)
                    
                    # drop unnecessary columns for display
                    rec_filtered = rec_filtered.drop(columns = ['weighted_avg', 'actors_downcased', 
                                                                'directors_downcased', 'title_downcased', 
                                                                'title_year', 'movieId', 'prediction',
                                                                'decade', 'tags_num'])
                        
                    # if no valid movies with combination of filters, notify. Else display dataframe
                    if len(rec_filtered) > 0:
                        st.write(rec_filtered)
                    else:
                        st.write('Found no recommended movies that match your selections')

