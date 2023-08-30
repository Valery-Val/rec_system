from catboost import CatBoostClassifier
import json
import ast
import os
from typing import List
import lightgbm as lgb

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

import pandas as pd
import numpy as np
import pickle
import catboost
import hashlib

### Connections etc.

SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
conn_full = engine.connect().execution_options(stream_results=False)

### Functions that we need

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features(query):
    return batch_load_sql(query)

### Colomns needed for the model in the right order

cols_order = ['gender',
 'country',
 'exp_group',
 'os',
 'source',
 'age_group',
 'is_capital',
 'cluster',
 'avg_tfidf',
 'part_of_likes',
 'topic',
 'text_lenght']

class PostGet:
    def __init__(self, id, text, topic):
        self.id = id
        self.text = text
        self.topic = topic

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model' # Need this to submit file to learning management system
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("model")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

rec_model = load_models() # Loading model
posts_info = load_features("SELECT * FROM public.v_elp_posts_with_embeddings") # Loading posts data

def is_capital(row):
    '''
    This function makes a new column with boolean value if the user's city is a capital city for their country.
    '''
    capital_cities = {
            'Switzerland': 'Bern',
            'Cyprus': 'Nicosia',
            'Latvia': 'Riga',
            'Estonia': 'Tallinn',
            'Finland': 'Helsinki',
            'Belarus': 'Minsk',
            'Kazakhstan': 'Nur-Sultan',
            'Azerbaijan': 'Baku',
            'Turkey': 'Ankara',
            'Ukraine': 'Kyiv',
            'Russia': 'Moscow'
        }
    return 1 if capital_cities[row['country']] == row['city'] else 0

def age_categorizer(row):
    '''
    This function defines user's age group base on psycological aspects and makes a new column.
    '''
    schoolers = 13 < row['age'] < 18 
    adolescents = 18 <= row['age'] < 24
    young = 24 <= row['age'] < 35 
    grown_ups = 35 <= row['age'] < 60
    wise =60 <= row['age']
    if schoolers:
        return 'schooler'
    elif adolescents:
        return 'adolescent'
    elif young:
        return 'young'
    elif grown_ups:
        return'grown_up'
    elif wise:
        return 'wise'
    else:
        return 'unknown'
    
###Functions for database acces

def get_user_by_id(user_id):
    user_data = pd.read_sql(f"SELECT * FROM public.v_elp_user_with_viewed_posts WHERE user_id = {user_id}", conn_full)
    viewed_posts_idx = user_data['array_agg'].iloc[0]
    user_data_net = user_data.drop('array_agg', axis = 1)
    user_data_net['age_group'] = user_data_net.apply(age_categorizer, axis=1)
    user_data_net['is_capital'] = user_data_net.apply(is_capital, axis=1)
    user_info = user_data_net.drop(['age', 'city'], axis = 1)

    return user_info, viewed_posts_idx

def get_posts_by_ids(posts_ids):
    query_string = "("
    for idx in posts_ids:
        query_string += str(idx)
        query_string += ","
    query_string = query_string[:-1]
    query_string += ")"
    posts = pd.read_sql(f"SELECT * FROM public.post_text_df WHERE post_id IN {query_string}", conn_full)
    return posts


### Function to get recommendations
def get_recomended_posts(user_id):
    user_info, viewed_posts_ids = get_user_by_id(user_id)
    new_posts = posts_info.loc[~posts_info.post_id.isin(viewed_posts_ids)]

    # Create a temporary key for merging
    user_info['key'] = 1
    new_posts['key'] = 1

    # Merge the two dataframes on the temporary key
    merged_df = pd.merge(user_info, new_posts, on='key')

    # Drop the temporary key
    merged_df.drop('key', axis=1, inplace=True)

    # Set 'user_id' and 'post_id' as multiindex
    merged_df.set_index(['user_id', 'post_id'], inplace=True)
    merged_df = merged_df[cols_order]

    # Get probabilities
    probs = rec_model.predict_proba(merged_df)
    merged_df.reset_index(level='user_id', inplace=True)

    # Get probabilities only for positive class
    could_be_liked = probs[:, 1]

    merged_df['like_probability'] = could_be_liked
    most_likely = merged_df.sort_values('like_probability', ascending = False).head(5)
    posts_idx = most_likely.index.tolist()
    to_propose = get_posts_by_ids(posts_idx)

    return to_propose

app = FastAPI()
@app.get("/post/recommendations/")
def recommended_posts(
		id: int, 
		limit: int = 10):
    recommended_posts_df =  get_recomended_posts(id)
    recommended_posts = []
    for _, row in recommended_posts_df.iterrows():
        post = PostGet(row['post_id'], row['text'], row['topic'])
        recommended_posts.append(post)

    return recommended_posts

# # app definition
# if __name__ == "main":
#     uvicorn.run(app, host="0.0.0.0", port=8000)~