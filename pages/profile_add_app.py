#!/usr/bin/env python
# coding: utf-8

# # Can't get object to be persistant
# 
# To Do: figure out how to save this profile for use in the personalized recommendations tab
# - Need to create a userId for them and append to original dataframe?
# - Or ask if they've create a profile? And then import a specific file that they've created? 

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


# In[ ]:


@st.cache(allow_output_mutation=True)
def list_create():
    return [], [], []


# In[ ]:


def write(df):
    
    st.title('Create a Ratings Profile')
    st.write('(1) Type the name of a movie you have watched')
    st.write('(2) Select a title from the dropdown of potentially matching movies in our system')
    st.write('(3) Provide a rating from 0.5-5.0 (higher better)')
    st.write('(4) Click **Submit** to submit this rating')
    st.write('Repeat as many times as desired')
    st.write('Click **Finished** to submit entire profile')
    
    
    # empty lists to hold user input: will persist across user refresh because of function
    ids, ratings, title = list_create()    
    
    # get user input text
    # too many movies for a full drop down 
    user_text = st.text_input("Enter a movie you have watched")
    # downcase input
    user_text = user_text.lower()

    if user_text == '':
        st.write('Waiting for input')
    else:

        # fuzzy string matching to find similarity ratio between user input and actual movie title (downcased)
        # works for misspellings as well 
        # limit to 70% similarity 
        options = df.copy()
        options['sim'] = options.title_downcased.apply(lambda row: fuzz.ratio(row, user_text))
        options = options[options.sim > 70].sort_values('sim', ascending = False).head(10).title_year.unique()

        # find movies with titles similar to what they typed
        if len(options) > 0:

            user_title = st.selectbox('Select Movie', ['<select>'] + list(options))

            # once input something, ask for rating
            if user_title != '<select>':

                # ask for rating
                user_rating = st.selectbox('Rate this Movie', [i/2 for i in range(1,11)])

                # once hit submit, add to data frame 
                if st.button('Submit'):

                    # find ID of movie they selected
                    user_movieid = int(df[df.title_year == user_title].movieId.values[0])

                    # add to persistant lists
                    ids.append(user_movieid)
                    ratings.append(user_rating)
                    title.append(user_title)
                    
                    
        # if nothing > 70% similiarity, then can't find a matching movie
        else:
            st.write("Sorry, we can't find any matching movies")
            
    if st.button('Finished'):
        # create dataframe from lists
        d = {'movieId':ids, 'title':title, 'rating':ratings}
        profile = pd.DataFrame(d)
        st.write('Here is your profile')
        st.write(profile)

