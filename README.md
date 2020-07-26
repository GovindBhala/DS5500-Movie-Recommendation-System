# DS5500-Movie-Recommendation-System

This project creates a movie recommendation system with a Streamlit UI to provide multiple types of users with tailored recommendations. Our ultimate goal is to increase viewership on a movie streaming platform and thus use these recommendations to help with the paradox of choice and to present new, "long tail" movies to users. Thus, we imagine this product being integrated into an existing streaming platform to utilize the platform's movie catalog and user's preferences through both explicit ratings and implicit clicks/views. To proxy this environment, we use the MovieLens dataset of user explicit ratings paired with IMDB movie metadata. The streamlit experience is intended to proxy a single login session in which users can view multiple types of recommendations and create their own personal profile if they do not have a pre-existing profile of rated movies. 

## Structure of Repo

## Data 

## UI Overview

## Models Overview
The two pages on our UI that are backed by models are "Movie Based Recommendations" and "Personalized Recommendations". We use different types of models in these scenarios based on the movie's attributes for Movie Based (Item-Item) or user's attributes for Personalized.   

![picture](images/model_flow.png)

The evaluation metrics used to select our final models are discussed in depth in the Methodological Appendix. At a high level, we want recommendations with the following attributes:
- Personalized recommendations: provide materially different sets of recommendations for different users
- Accurate recommendations: high precision and recall based on test/train split of user ratings
- Personal diversity: provide variety of recommendations to each individual user
- Average rating: recommend high quality movies with high average ratings
- Global diversity: recommend movies in the long tail. Do not only recommend popular movies because this will not increase overall viewership, engagement with the streaming platform 

To achieve all of these goals, we use combinations of several models to produce recommendations. 
- Content-Based Models
- Personalized Models  -- discuss pros and cons of these kinds of models in general and then our specific models and their pros/cons 

# Methodological Appendix

## Evaluation Metrics 

### Personalization
_Goal_: Maximize differences between move recommendations for different people
_Method_: K fold cross-validation across several sets of users
![picture](images/personalization.PNG)

### Precision, Recall @ K
- _Goal_: Maximize "accuracy" of model i.e. ability to retrieve recommendations that user's actually like
- _Method_: 
  - Split data into train/test sets by selecting random users and then for each user, splitting their ratings half into test and half into train. 
  - Generate recommendations based on training data. See if get movies from the test data that the user actually liked    
  - Ideally would have real time user feedback, for the recommendations we produce to assess accuracy. Test/train split is a proxy in lieu of that data. It is very difficult, especially for content based recommendation systems, to achieve good precision and recall because we are generating predictions for all of the ~45.000 in the catalog and then recommending 10. Unlikely that the user has rated those 10 movies in the test set. Does not mean the recommendations are bad. 
  
![title](images/precision_recall.PNG)

### Personal Diversity 
- _Goal_: For individual users, maximize the variety of movies that are recommended
  - Content models tend to create overspecialization where users are only presented with one type of movie. Thus need to evaluate the degree of this problem. 
  - Filtering function in UI also helps deal with this problem as users can view specific types of recommendations   
- _Method_:
  - Find cosine similarity between recommended movies for a particular user
  - Movie features depend on the model being evaluated (if tags model, look at diversity of tags. If genre, actors model, look at diversity of genres and actors)
![title](images/personal_diversity.PNG)

### Average Rating
- _Goal_: Recommend "good" movies with high average ratings. Recommendation system is not specifically designed for this, but good to track across systems   
- _Method_: 
![title](images/avg_rating.PNG)

### Global Diversity
- _Goal_: Recommend some "unpopular" movies such that users view movies in the long tail that they otherwise would not be exposed to. Thus increase overall viewership. 
- _Method_: Minimize this metric 
![title](images/global_diversity.PNG)

## Model Iteration & Performance 
