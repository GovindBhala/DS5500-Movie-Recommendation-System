#!/usr/bin/env python
# coding: utf-8

# # Create New Profile
# - Persistant across session: if come back to page later (without clearing cache), will be on the same profile
# - NOT saved for next session
# - Add to LISTS instead of dataframes because mutable: persistant across sessions 
#     - Thus can enter ID created in personalization page to get recommendations    
#       
# __Questions:__
# - Give option to make a new profile? Not sure how to do this     
# - Save so persitant between sessions?

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


# function creates empty lists, but allow output mutation such that not overwritten when page refreshes
# only works with mutable data types
@st.cache(allow_output_mutation=True)
def list_create():
    return [], [], [], []


# In[19]:


def write(df, ratings_lst, user_lst, movies_lst):
    
    # empty lists to hold user input: will persist across user refresh because of function
    ids, ratings, title, userId_new = list_create()    

    # generate a new user id 
    # append to list because changes every time the page is run. Only want first max entry. 
    userId_new.append(int(np.max(user_lst) + 1))
    
    st.title('Create a Ratings Profile')
    st.write('(1) Type the name of a movie you have watched')
    st.write('(2) Select a title from the dropdown of potentially matching movies in our system')
    st.write('(3) Provide a rating from 0.5-5.0 (higher better)')
    st.write('(4) Click **Submit** to submit this rating. Repeat as many times as desired')
    st.write('(5) Click **View Profile** to view your profile')
    st.write('Enter your user ID on the Personalized Recommendation pages. Feel free to return and add more movies later.')
    st.write('')
    st.write('**Your User ID is: ' + str(userId_new[0]) + '**')
        
    # get user input text - too many movies for a full drop down 
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

                # once hit submit, add to lists
                if st.button('Submit'):

                    # find ID of movie they selected
                    user_movieid = int(df[df.title_year == user_title].movieId.values[0])

                    # add to persistant lists for this profile
                    ids.append(user_movieid)
                    ratings.append(user_rating)
                    title.append(user_title)
                    
                    # add to overall lists: use in personalized recs
                    ratings_lst.append(user_rating)
                    user_lst.append(userId_new[0])
                    movies_lst.append(user_movieid)
                
                    
        # if nothing > 70% similiarity, then can't find a matching movie
        else:
            st.write("Sorry, we can't find any matching movies")
            
    # view your profile 
    if st.button('View Profile'):
        # create dataframe from lists and display entered profile
        d = {'movieId':ids, 'title':title, 'rating':[str(round(i, 1)) for i in ratings]}
        profile = pd.DataFrame(d)
        st.write('Here is your profile')
        st.write(profile)
                  
            
            
        # print generated userId for user to use in personalized recs
        #userId_new = ratings_df.userId.max() + 1 # generate a new user id 
        
        #### originally tring to save so persistant across sessions. May return to this. 
        #ratings_lst.append(ratings)
        #user_lst.append([userID_new]*len(ratings))
        
        # append to ratings data and save so can load into personalization 
        #d = {'movieId':ids, 'rating':ratings, 'timestamp': [None]*len(ids), 'userId': [userId_new]*len(ids)}

        #ratings_df = pd.concat([ratings_df, pd.DataFrame(d)])
        #st.write('Saving your profile. Please wait...')
                 
        # save for future loads of app
        #ratings_df.to_parquet('ratings_sample_useradd.parq', engine = 'fastparquet', compression = 'GZIP', index = False)
        
        #st.write('Done! Please enter your ID number in the Personalized Recommendations page')    
        
    # return so can be used in this current run
    return ratings_lst, user_lst, movies_lst

