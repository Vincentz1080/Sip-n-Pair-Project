import pandas as pd
import pickle
import re
import json
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import nltk

print("--- Starting Environment Setup ---")
nltk.download('stopwords', quiet=True)

stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

def clean(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return ' '.join([stemmer.stem(w) for w in text.split()
                     if w not in stop_words and len(w) > 2])

def main():
    print("--- Starting preprocessing ---")

    # --- Wine Index ---
    print("Processing wine reviews...")
    wine_df = pd.read_csv('data/wine_reviews.csv').dropna(subset=['description'])
    wine_df = wine_df.reset_index(drop=True)
    wine_df['text'] = (wine_df['description'].fillna('') + ' ' +
                       wine_df['variety'].fillna('') + ' ' +
                       wine_df['region_1'].fillna('')).apply(clean)

    print("Vectorizing Text...")
    wine_vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    wine_tfidf = wine_vec.fit_transform(wine_df['text'])

    print("Fitting TruncatedSVD for wines...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    wine_svd = Normalizer().fit_transform(svd.fit_transform(wine_tfidf))

    terms = wine_vec.get_feature_names_out()
    print("\nTop terms per SVD dimension (first 10 dimensions):")
    for i, component in enumerate(svd.components_[:10]):
        top = [terms[j] for j in component.argsort()[-8:][::-1]]
        print(f"Dim {i}: {top}")

    # --- Food Index ---
    print("\nProcessing food recipes (loading first 40k rows to optimize memory)...")
    food_df = pd.read_csv('data/food_recipes.csv', nrows=40000).dropna(subset=['name'])
    food_df = food_df.reset_index(drop=True)
    food_df['text'] = (food_df['name'].fillna('') + ' ' +
                       food_df['ingredients'].fillna('') + ' ' +
                       food_df['tags'].fillna('')).apply(clean)

    food_vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    food_tfidf = food_vec.fit_transform(food_df['text'])

    # --- FlavorDB Index ---
    print("\nProcessing FlavorDB (JSON Format)...")
    flavor_index = {}
    with open('data/flavordb2.json', 'r', encoding='utf-8') as f:
        flavor_data = json.load(f)

    for item in flavor_data:
        ingredient = str(item.get('entity_alias_readable', '')).lower().strip()
        molecules = item.get('molecules')
        if not ingredient or not isinstance(molecules, list):
            continue
        
        for molecule in molecules:
            flavor_str = molecule.get('flavor_profile')
            if flavor_str and isinstance(flavor_str, str):
                # JSON format uses '@' to separate flavors, not ','
                flavors = flavor_str.lower().split('@')
                for f in flavors:
                    f = f.strip()
                    if f:
                        flavor_index.setdefault(f, []).append(ingredient)

    # Deduplicate ingredients for each flavor to speed up search later
    for f in flavor_index:
        flavor_index[f] = list(set(flavor_index[f]))

    print(f"Extracted {len(flavor_index)} unique flavor categories from FlavorDB.")

    # --- Save everything ---
    print("\nSaving models and processed data to disk...")
    pickle.dump(wine_tfidf,    open('wine_tfidf.pkl', 'wb'))
    pickle.dump(wine_vec,      open('wine_vec.pkl', 'wb'))
    pickle.dump(wine_svd,      open('wine_svd.pkl', 'wb'))
    pickle.dump(svd,           open('svd_model.pkl', 'wb'))
    pickle.dump(food_tfidf,    open('food_tfidf.pkl', 'wb'))
    pickle.dump(food_vec,      open('food_vec.pkl', 'wb'))
    pickle.dump(flavor_index,  open('flavor_index.pkl', 'wb'))
    wine_df.to_csv('wine_clean.csv', index=False)
    food_df.to_csv('food_clean.csv', index=False)

    print("Preprocessing complete!")

if __name__ == '__main__':
    main()
