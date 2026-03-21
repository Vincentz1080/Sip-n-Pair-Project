"""
Routes: home page and disease search.
"""
import json
import os
import pandas as pd
import numpy as np
from flask import render_template, request

# Load dataset once
basedir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(basedir, 'compact_dataset.csv')
df = pd.read_csv(dataset_path)

# Create a list of symptoms (excluding the 'diseases' column)
symptoms = df.columns[1:].tolist()

def json_search(query):
    if not query or not query.strip():
        return json.dumps([])
    
    query = query.lower()
    
    # 1. Extract keyword matches
    user_vector = []
    matched_symptoms_list = []
    for sym in symptoms:
        if sym.lower() in query:
            user_vector.append(1)
            matched_symptoms_list.append(sym)
        else:
            user_vector.append(0)
            
    # 2. Similarity calculation (dot product between user symptoms and disease matrix)
    user_vec_np = np.array(user_vector)
    
    if sum(user_vector) == 0:
        return json.dumps([])
        
    disease_matrix = df.iloc[:, 1:].values
    scores = disease_matrix.dot(user_vec_np)
    
    # 3. Filter and rank illnesses
    df_scored = df.copy()
    df_scored['score'] = scores
    
    top_results = df_scored[df_scored['score'] > 0].sort_values(by='score', ascending=False).head(10)
    
    matches = []
    for _, row in top_results.iterrows():
        matches.append({
            'disease': str(row['diseases']).title(),
            'score': int(row['score']),
            'matched_symptoms': ", ".join(matched_symptoms_list)
        })
    return json.dumps(matches)

def register_routes(app):
    @app.route("/")
    def home():
        return render_template('base.html')

    @app.route("/search")
    def search_endpoint():
        text = request.args.get("query", "")
        return json_search(text)
