### Recommendation system for social network
This repository contains a part of final project for courses I attended. 
The task was to build recommendation system for social network which would suggest 5 posts individually for a user.
The project consists of two parts:
1. Jupyter notebook with all the steps of training model.
2. Service to give recommendations.

#### Training model
Jupyter notebook ('training_model.ipynb') contains all the steps including both EDA and training part.
Feature engineering part includes using language model (BERT) to create embeddings for post texts.
CatBoost model is used to predict users' reaction to particular posts.

#### Backend service
All data on posts is downloaded at the time of application launch.
The model is used to predict reaction of a particular user on new posts (the ones they have not seen by the time the application was launched).
The system gets 5 posts with highest probability of the first class (meaning the user will like the post).

#### Run the application
1. Make sure you have all the necessary libraries installed. Check 'requirements.txt' file.
2. File with model is attached.
3. To run the application uvicorn is used:
   
   ```
   cmd /C "set DATABASE_URL=<url to connect the database>&& set MODEL_PATH=<path to model>&& uvicorn app:app --port 8899"
   ```
