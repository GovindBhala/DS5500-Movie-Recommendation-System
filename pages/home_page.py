#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st


# In[ ]:


def write():
    st.title('Welcome to the Movie Recommender!')
    st.header('Select the type of recommendation you want on the left menu')
    st.write("""
    Options:
    - **Top Movie Visualizations**: view visualizations of the movies in our catalog to understand what makes a great movie.
    - **Top Rated Movies**: apply filters to find the top rated movies in your desired attributes
    - **Movie Based Recommendations**: enter a movie you have previously enjoyed to view similar movies
    - **Personalized Recommendations**: enter your user ID to find your personalized top movies + apply filters
    - **Add Profile**: manually enter movie ratings if you are not in our system, thereby creating a profile to enable personalized recommendations
    """)

