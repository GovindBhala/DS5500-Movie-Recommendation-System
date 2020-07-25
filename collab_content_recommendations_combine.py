#!/usr/bin/env python
# coding: utf-8

# # Combine collaborative filtering and content recommendation results
# 
# - Produce half recommendations from collaborative filtering and half from the content model (half based on specified number of recommendations top_n)
# - Content model recommendations out of movies not included in collaborative filtering recommendations _for that user_ 
# - Re-sort based on similarity scores 
#     - Need to redo this to sort on weighted average of movie ratings such that we produce the most "credible"/recognizable results first to gain the user's trust before presenting long tail recommendations
#     - And here the similarity scores are entirely different. From collab filtering, it is predicted rating (0.5-5) and from content it is cosine similarity 
# 
# Parameters:
# - user_id: ID of user to generate recommendations for
# - df1: sparse matrix of movie attributes in one hot encoded fashion with attributes from for content model 
# - ratings: ratings data for each user (movies rated + star ratings)
# - movieIds: list of all movie Ids (rows of sparse matrix)
# - keep_movies1: [] -- dummy parameter so that this funciton as the same inputs as the other recommendation models
# - collab_predictions: pregenerated collaborative filtering predictions. Equivalent of df2 in other recommendation model parameters
# - keep_movies2: [] -- dummy parameter so that this funciton as the same inputs as the other recommendation models
# - content_recommendation_system: recommendation system to use to generate recs for content model
#     - Module of a function in another script
# - top_n: number of recommendations total to produce

# In[121]:


import pandas as pd
import os
import numpy as np
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


# In[82]:


# not using keep_movies1 or keep_movies2 
# collab_predictions as df2
def collab_content_combine(user_id, df1, ratings, movieIds, movies_ratings, keep_movies1, collab_predictions,
                           keep_movies2, content_recommendation_system, top_n = 10):
    
    # get recommendations from collab filtering model 
    collab_rec = collab_predictions[collab_predictions.userId == user_id]
    collab_rec = pd.merge(collab_rec, movies_ratings, on = 'movieId')
    collab_rec = collab_rec.sort_values(['prediction', 'weighted_avg'], ascending = [False, True])

    # find movies in full set that are not in collaborative filtering predictions for this user
    keep_movies = set(movieIds).difference(set(collab_rec.movieId.unique()))
    
    # generate recommendations from content model with movies not in colalb filtering
    content_rec = content_recommendation_system(user_id, df1, ratings, movieIds, movies_ratings, keep_movies)
    
    # concat half top recommendations from each model 
    recommendations = pd.concat([collab_rec.head(int(top_n/2)), content_rec.head(int(top_n/2))])
    
    # resort based on similarity scores
    recommendations = recommendations.sort_values('weighted_avg', ascending = False)
    
    return recommendations[['movieId', 'prediction']]

