# DS5500-Movie-Recommendation-System

This project creates a movie recommendation system with a Streamlit UI to provide multiple types of users with tailored recommendations. Our ultimate goal is to increase viewership on a movie streaming platform and thus use these recommendations to help with the paradox of choice and to present new, "long tail" movies to users. Thus, we imagine this product being integrated into an existing streaming platform to utilize the platform's movie catalog and user's preferences through both explicit ratings and implicit clicks/views. To proxy this environment, we use the MovieLens dataset of user explicit ratings paired with IMDB movie metadata. The streamlit experience is intended to proxy a single login session in which users can view multiple types of recommendations and create their own personal profile if they do not have a pre-existing profile of rated movies. 

## Structure of Repo

## Data 
This project utilizes two public datasets. 

1. __MovieLens__: movie rating data collected from the MovieLens application created by the GroupLens Research lab at the University of Minnesota. MovieLens provides non-commercial, personalized recommendations to users for free. We are using the 25 Million dataset, which includes ~58,000 movies with ratings from ~28M users. Ratings are from 0.5 to 5 stars with 0.5 increment. GroupLens also produces genome tags using a machine learning algorithm based on user inputted tags, ratings, and text reviews. Each movie is given a relevance score (0-1) for each tag.             
Dataset: https://grouplens.org/datasets/movielens/            
MovieLens Website: https://movielens.org/          

2. __IMDb movies extensive dataset__: Scraped data from the IMDB website including release year, genre, duration, director(s), actors, production company, language, country, and description (and more).          
Dataset (Kaggle): https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset            

We merge these two datasets on each movie's IDMB ID, which is provided in both datasets. We drop about 17,000 movies in this merge as the IMDB set does not include all of the movies in the MovieLens set. 
 
As described in the modeling section, some of our models are content-based and thus depend on this movie metadata. We ultimately one-hot-encode all of the features used to build the models, but first perfrom the following feature engineering.
- Both MovieLens and IMDB include, sometimes non-matching, genre lists. We take the union of both lists for each movie. 
- Limit the actors lists to the 3 top actors in each movie. We cannot reasonably one-hot-encode all actors for all movies due to memory and performance limitations. Additionally, minor actors likely do not affect a user's movie preference. 
  - Top actors are determined by the number of movies in the catalog they have appeared in 
- Find top TF-IDF scoring tokens for a variety of text fields: MovieLens genome tags, movie description, and combined tags + description field 
  - First clean these fields: downcase, remove non-ASCII, remove punctuation, tokenize, remove stop words (leave in some key negation words), and lemmatize
- Drop all categorical values that appear in only one movie. For example, actors in only one movie. These are not helpful for finding similar movies and increase memory, performance needs

## UI Overview
Our UI is built on Streamlit (https://www.streamlit.io/)    
It consists of 5 pages:
1. Top Movie Visualizations: View visualizations of the movies in our catalog to understand what makes a great movie
 - Subset of visualizations created in exploratory data analysis that would be informative to a user understanding how to choose a good movie to watch
 - Use the checkboxes to view/close each visualization
2. Top Rated Movies: Apply filters to find the top rated movies with your desired attributes
 - Provide recommenations to non-users. Simple user inputs and filtering, no underpinned model
 - Movies are sorted on the weighted average between their average rating and total number of ratings. If we sort on average rating only, there are many movies that are sorted only a few times and thus have unrealistically high average ratings 
3. Movie Based Recommendations: Enter a movie that you have previously enjoyed to view similar movies
 - Item-Item recommendation model. Find movies similar to the inputted movie based on its content profile
4. Personalized Recommendations: Enter your user ID to find your personalized top movies + apply filters
 - Personalized movies based on specific user. Uses a combination of user-user collaborative filtering with KNN and a content-based model
 - User can filter down top recommendations 
5. App Profile: If you are not in our system, create a new profile to enable personalized recommendations
 - Generates a unique user ID
 - User can enter movies and ratings to generate a new profile that can then be entered into the Personalized Recommendations tab


## Models Overview

The __Evaluation Metrics__ used to select our final models are discussed in depth in the Methodological Appendix. At a high level, we want recommendations with the following attributes:
- Personalized recommendations: provide materially different sets of recommendations for different users
- Accurate recommendations: high precision and recall based on test/train split of user ratings
- Personal diversity: provide variety of recommendations to each individual user
- Average rating: recommend high quality movies with high average ratings
- Global diversity: recommend movies in the long tail. Do not only recommend popular movies because this will not increase overall viewership, engagement with the streaming platform 

To achieve all of these goals, we use combinations of several models to produce recommendations. At the highest level, there is a trade-off between content based models and collaborative filtering models.          
              
__Content-Based Models__: recommend movies with similar meta-data attributes to movies that the user had rated highly              
Pros:
  - Gobal diversity: no cold start problem - can recommend new and unpopular movies 
  - Personalization: recommendations are not biased towards a smaller subset of popular movies. Content models will recommend a wide variety of movies, thus increasingly the likelihood of generating different recommendations for different users    
              
Cons:
  - Personal diversity: over-specialization when defining a user's profile such that they are recommended a narrow set of very similar movies
  - Accurate rating: it is very difficult to achieve good precision and recall because we are generating predictions for all of the ~45,000 movies in the catalog and then recommending 10. Unlikely that the user has rated those 10 movies in the test set. Does not mean the recommendations are bad.            
             
__Collaborative Filtering Models__: recommend movies that were rated highly by users that have a similar rating history to the user                
Pros:
  - Average rating: biased towards popular movies which means both frequently watched and highly rated movies
  - Accurate recommendations: users are more likely to rate popular movies and collaborative filtering models are more likely to recommend popular movies. Thus more likely than content-based to get a match between recommended and test set.          
            
Cons: 
  - Personalization: biased towards a smaller subset of popular movies, so hard to generate different recommendations for different users
  - Global diversity: biased towards popular movies that have been watched many times. Most implementations explicitely exclude movies with small numbers of ratings. 
  
Our __final personalized model__ uses a combination of collaborative filtering and content-based approaches. For movies with more than 50 ratings, we use a KNN user-user collaborative filtering method to find similar users. For all other movies, we use a content-based approach that finds similar movies based on their genres, actors, and directors. For the final recommendation list, we take the top 5 movies from each system and present a list of 10 movies sorted by the weighted average between the movie's number of ratings and average rating. We thus recommend movies that we are confident the user will like through collaborative filtering and movies in the long tail through content-based to achieve all of our goals. We sort on weighted average because we want to present the most popular movies first in order to gain the user's trust, and then present the less popular "long-tail" movies that they likely have not heard of in hopes of increasing our platform's overall number of streams.   

Our app also provides movie based recommendations where the user inputs a movie and we recommend similar movies. The __final item-item model__ uses a purely content-based approach because we do not have user rating data in this instance. The final content model is a combination of two content profiles. For movies with genome tags, we find similar movies based on their top 5 TF-IDF tokens from a combined text field of genome tags plus movie description. For movies without genome tags, we find similar movies based on their genres, actors, and directors. For the final recommendation list, we take the top 5 movies from each system and, again, present a list of 10 movies sorted by the rating weighted average. Movies with tags are generally more popular with more ratings than movies without tags. The recommendations based on tags have better precision and recall than recommendations based on genre, actors, and directors, but they fail to reach the long tail. Thus, similar to the personalized model, we present a mix of confident recommendations and long-tail recommendations.

There are two caveats to these two final models: 
1. If a user enters a new profile in the app during a session, we use the content-only based approach that we use for item-item. The collaborative filtering model is precomputed and requires training to generate recommendations. The content model meanwhile generates recommendations on demand. The new profile is saved and will be included in periodic collaborative retraining such that those recommendations would be available in the future. 
2. If the movie entered on the 'Movie Based Recommendations' page does not have any genome tags, our recommendations will only be based on genres, actors, and directors. We cannot use tags to recommend if the reference movie does not have tags. 

__Summary of the Model Flow__    
These flow diagrams represent the two model based pages in the UI.

![picture](images/model_flow.png)


# Methodological Appendix

## Evaluation Metrics 
- For all metrics, calculated on a random subset of users and average value taken across users. For personalization, take average across K sets (folds) of randm users because comparing userse to each other
- Make sure to set the seed such that different models are evaluated on the same random subset 
- Results from all models are recorded in 'evaluations' folder as text files 

### Personalization
- _Goal_: Maximize differences between move recommendations for different people
- _Method_: K fold cross-validation across several sets of users
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
