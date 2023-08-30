## Recommendation System for Social Networks
This repository is a segment of the final project from the courses I undertook. It's focused on developing a recommendation system for social networks. The primary goal of this system is to suggest five personalized posts to individual users. The project is divided into two primary components:

1. A Jupyter notebook detailing all the model training steps.
2. A service that offers recommendations.
   
### Training Model
The Jupyter notebook, titled `training_model.ipynb`, covers all the necessary steps, encompassing Exploratory Data Analysis (EDA) and model training. The feature engineering segment leverages a language model (specifically BERT) to generate embeddings for post texts. For predicting users' reactions to particular posts, the CatBoost model is utilized.

### Backend Service
Upon launching the application, all post data is fetched. Using the trained model, the system predicts a user's potential reactions to new posts (posts they haven't encountered at the time of application launch). The system then retrieves the top five posts with the highest likelihood of receiving a positive reaction (classified under the first class, which indicates the user will appreciate the post).

### Getting Started
#### Prerequisites
Ensure you have all the essential libraries installed. You can refer to the `requirements.txt` file for a comprehensive list. The model file is also included in the repository.

#### Launching Jupyter Notebook
Use the following command to launch Jupyter Notebook from terminal:

```
cmd /C "set DATABASE_URL=<URL to your database>&& set MODEL_PATH=<path to your model>&& jupyter notebook"
```

#### Launching the Application
The application utilizes uvicorn for execution. Use the following command to run the app:

```
cmd /C "set DATABASE_URL=<URL to your database>&& set MODEL_PATH=<path to your model>&& uvicorn app:app --port 8899"
```
