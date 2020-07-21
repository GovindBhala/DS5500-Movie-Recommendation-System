#!/usr/bin/env python
# coding: utf-8

# # Item-Item Recommendations for Non-Users (Similar Movies)
# **Use Case**: User without a profile can get recommendations based on a movie they have previously enjoyed by displaying similar movies.    
# Details:
# - Allow users to put in a movie that they liked and output similar movies 
# - Use same content profiles as in personalized recommendations:
#     - For movies with genome tags, find recommendations based on similarity with top tf-idf tokens of description+tags field
#     - 
# - Sort on weighted average: present most popular movies at the top to gain credibility, and then present long tail movies to generate more streaming after have gained trust  
#     
# Process:

# In[1]:


import pandas as pd
import numpy as np
import re
import streamlit as st
from fuzzywuzzy import fuzz
import scipy
import pickle


# ## Item-Item Recommendations

# In[14]:


@st.cache(allow_output_mutation = True)
def item_recs(df, df_display, movieIds, user_movieId, keep_movies):
    
    # get profile of selected movie
    selected_movie_index = movieIds.index(user_movieId)
    selected_movie = df[selected_movie_index,:]
    
    # similarity with all movies: result is sum of similar features 
        # ex 9 = 9 identical features. 0 = no identical features
    recommendations = pd.DataFrame(df.dot(selected_movie.T).todense())
    
    # merge similarities with movieIds
    recommendations = pd.merge(recommendations, pd.Series(movieIds).to_frame(), left_index = True, right_index = True)
    recommendations.columns = ['prediction', 'movieId']

    # sort and merge with display data
    recommendations = recommendations.sort_values('prediction', ascending = False)
    recommendations = pd.merge(recommendations, df_display, on = 'movieId', how = 'left')
    
    # remove entered movie
    recommendations = recommendations[recommendations.movieId != user_movieId]
    
    # remove movies not in keep_movies options
    recommendations = recommendations[recommendations.movieId.isin(keep_movies)]

    return recommendations


# In[15]:


@st.cache(allow_output_mutation = True)
def item_recs_combined(df1, df2, df_display, movieIds, user_movieId, keep_movies1, keep_movies2, top_n = 10):
    
    # recommendations from each model 
    recs_notags = item_recs(df1, df_display, movieIds, user_movieId, keep_movies = keep_movies1)
    recs_tags = item_recs(df2, df_display, movieIds, user_movieId, keep_movies = keep_movies2)
    
    # concat half top recommendations from each model 
    recommendations = pd.concat([recs_notags.head(int(top_n/2)), recs_tags.head(int(top_n/2))])

    # resort based on weighted average: put most popular movies at the top to gain credibility 
    recommendations = recommendations.sort_values('weighted_avg', ascending = False)
    
    return recommendations 


# # Streamlit App with User Input
# - User input: text input
# - Find options that are close to the text input based on fuzzy string matching
#     - Works for not-quite-right movie title and misspellings
#     - Top 10 IF similarity ratio > 70. Don't display anything if similarity too low
# - Select out of drop down. Drop down includes (year) 
#     - Some titles are duplicate so need year
# - Get movieId from selection and use that as recommendation input

# In[17]:


def write(df_display, df1, df2, movieIds, movieIds_notags, movieIds_tags):

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
        options = df_display.copy()
        options['sim'] = options.title_downcased.apply(lambda row: fuzz.token_sort_ratio(row, user_text))
        options = options[options.sim > 70].sort_values('sim', ascending = False).head(10).title_year.unique()

        # find movies that are similar to what they typed 
        if len(options) > 0:

            # select from dropdown 
            user_title = st.selectbox('Select Movie', options)
            # get ID of selected movie
            user_movieid = df_display[df_display.title_year == user_title].movieId.values[0]

            if st.button('Display Recommendations'):
                
                # if selected movie has genome tags, 
                    # use the combined model with 5 recs based on popular movies with tags and 5 form long tail w/o tasg
                # if selected movie does not have genome tags, 
                    # generate all 10 recommendations from metadata based model (genre, actor, director) w/ no limits on movies
                if df_display[df_display.movieId == user_movieid].tags_num.values[0] > 0:
                    recs = item_recs_combined(df1, df2, df_display, movieIds, user_movieid, movieIds_notags, movieIds_tags)
                else:
                    recs = item_recs(df1, df_display, movieIds, user_movieid, movieIds)
                    recs = recs.head(10)
                    recs = recs.sort_values('weighted_avg', ascending = False)

                # display
                st.write(recs.drop(columns = ['movieId', 'weighted_avg', 'actors_downcased', 'directors_downcased',
                                              'title_downcased', 'title_year', 'decade', 'prediction', 'tags_num']))

        # if nothing > 70% similiarity, then can't find a matching movie
        else:
            st.write("Sorry, we can't find any matching movies")

