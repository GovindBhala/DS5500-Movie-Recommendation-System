#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
import sklearn
import fastparquet
import scipy


# ## Content Recommendations
# Extensions:
# - Include year (decade)?
# - Include text from description or genome tags
# - Downweight older ratings 

# ### Version 2
# Do not normalize ratings vectors. Only normalize profile. Then multiply matricies
# Previously taking cosine similarity and thus normalizing both vectors. This caused longer movie vectors to be penalized, thereby prioritizing movies without any actor and/or director dummies because they were shorter.    
# 
# Normalize profile so take into account different # of ratings for different users.   
# No need to normalize movies because 0/1 values and having more features is not a negative.  

# In[ ]:


def user_content_recommendations(user_id, df, ratings, movieIds):   
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
   
    recommendations = metrics.pairwise.cosine_similarity(df, profile)
    recommendations = pd.DataFrame(recommendations)
    recommendations = pd.merge(recommendations, pd.Series(movieIds).to_frame(), left_index = True, right_index = True)
    recommendations.columns = ['prediction', 'movieId']
    recommendations = recommendations[~recommendations.movieId.isin(watched)]
    recommendations = recommendations.sort_values('prediction', ascending = False)
    #recommen_ratings = pd.merge(recommendations,movies_raitings, left_on = 'movieId', right_on = 'id')
    return recommendations

