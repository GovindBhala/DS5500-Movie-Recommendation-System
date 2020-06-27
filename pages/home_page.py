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
    - Top Rated Movies: apply filters to find the top rated movies in your desired attributes
    - Movie Based Recommendations: enter a movie you have previously enjoyed to view similar movies
    """)

