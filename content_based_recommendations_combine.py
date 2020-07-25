#!/usr/bin/env python
# coding: utf-8

# # Combine two content recommendation results 
# Produce half recommendations from one model and half from the other (half based on specified number of recommendations top_n)
# Re-sort based on similarity scores 
# - May redo this to sort on weighted average of movie ratings such that we produce the most "credible"/recognizable results first to gain the user's trust before presenting long tail recommendations
# 
# 
# Parameters:
# - user_id: ID of user to generate recommendations for
# - df1: sparse matrix of movie attributes in one hot encoded fashion with attribute set 1 for model 1 
# - ratings: ratings data for each user (movies rated + star ratings)
# - movieIds: list of all movie Ids (rows of sparse matrix)
#     - Same for both models as df1 and df2 include all movies to generate user profiles from 
# - keep_movies1: subset of movies (list of movie ids) that we want to limit our recommendations to for model 1
# - df2: sparse matrix of movei attributes in one hot encoded fashion with attribute set 2 for model 2
# - keep_movies2: subset of movies (list of movie ids) that we want to limit our recommendations to for model 2
# - recommendation_system: recommendation system to use to generate recs for both model 1 and 2 
#     - Module of a function in another script
# - top_n: number of recommendations total to produce

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
import pickle
import scipy
import sklearn
import content_based_recommendations


# In[ ]:


def content_models_combine(user_id, df1, ratings, movieIds, movies_ratings, 
                           keep_movies1, df2, keep_movies2, recommendation_system, top_n = 10):
    
    # generate recommendations from each model with respective dataframes and kept movies
    recommend1 = recommendation_system(user_id, df1, ratings, movieIds, movies_ratings, keep_movies1)
    recommend2 = recommendation_system(user_id, df2, ratings, movieIds, movies_ratings, keep_movies2)
    
    # concat half top recommendations from each model 
    recommendations = pd.concat([recommend1.head(int(top_n/2)), recommend2.head(int(top_n/2))])
    
    # resort based on similarity scores
    recommendations = recommendations.sort_values('prediction', ascending = False)
    
    return recommendations

