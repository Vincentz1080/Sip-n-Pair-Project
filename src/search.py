import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

# Import our cleaning function
from preprocess import clean

# Load models and data once globally
wine_tfidf   = pickle.load(open('wine_tfidf.pkl', 'rb'))
wine_vec     = pickle.load(open('wine_vec.pkl', 'rb'))
wine_svd     = pickle.load(open('wine_svd.pkl', 'rb'))
svd_model    = pickle.load(open('svd_model.pkl', 'rb'))
food_tfidf   = pickle.load(open('food_tfidf.pkl', 'rb'))
food_vec     = pickle.load(open('food_vec.pkl', 'rb'))
flavor_index = pickle.load(open('flavor_index.pkl', 'rb'))
wine_df      = pd.read_csv('wine_clean.csv')
food_df      = pd.read_csv('food_clean.csv')

# Expanded common wine flavor descriptors
FLAVOR_KEYWORDS = [
    'blackberry', 'cherry', 'plum', 'raspberry', 'strawberry',
    'citrus', 'lemon', 'apple', 'peach', 'vanilla', 'chocolate',
    'coffee', 'tobacco', 'oak', 'smoke', 'earth', 'floral',
    'pepper', 'honey', 'butter', 'mushroom', 'fig', 'caramel',
    'leather', 'mineral', 'spice', 'cedar', 'violet', 'rose'
]

def search_wines(query, use_svd=True, top_k=10):
    """
    Match user wine description against wine review corpus.
    Uses TruncatedSVD dimensionality reduction if use_svd=True.
    Otherwise uses pure TF-IDF.
    """
    q = clean(query)
    q_tfidf = wine_vec.transform([q])

    if use_svd:
        q_vec = Normalizer().fit_transform(svd_model.transform(q_tfidf))
        sims = cosine_similarity(q_vec, wine_svd).flatten()
    else:
        sims = cosine_similarity(q_tfidf, wine_tfidf).flatten()

    idx = sims.argsort()[-top_k:][::-1]
    results = wine_df.iloc[idx].copy()
    results['score'] = sims[idx]
    results['idx'] = idx
    return results

def extract_flavors(wine_results, top_n=5):
    """
    Counts frequency of flavor descriptors in the top retrieved wine reviews.
    Returns the most prominent flavors as a sorted list.
    """
    counts = {}
    for desc in wine_results.head(top_n)['description']:
        desc = str(desc).lower()
        for f in FLAVOR_KEYWORDS:
            if f in desc:
                counts[f] = counts.get(f, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)[:8]

def get_complementary_ingredients(flavors, top_n=15):
    """
    Bridging logic: Query FlavorDB for ingredients that map to the identified wine flavors.
    """
    scores = {}
    for flavor in flavors:
        for key, ingredients in flavor_index.items():
            if flavor in key or key in flavor:
                for ing in ingredients:
                    scores[ing] = scores.get(ing, 0) + 1
    return sorted(scores, key=scores.get, reverse=True)[:top_n]

def search_foods(ingredients, top_k=8):
    """
    Matches complementary Food ingredients into actual recipes
    from our Food.com database using TF-IDF cosine similarity.
    """
    if not ingredients:
        return pd.DataFrame(columns=['name', 'description', 'score'])
        
    query = ' '.join(ingredients)
    q_vec = food_vec.transform([clean(query)])
    sims = cosine_similarity(q_vec, food_tfidf).flatten()
    idx = sims.argsort()[-top_k:][::-1]
    
    results = food_df.iloc[idx].copy()
    results['score'] = sims[idx]
    return results

def explain_svd(query, top_n=3):
    """
    Returns the most activated SVD latent dimensions (both Positively and Negatively)
    for grading exposition in the UI.
    """
    q_tfidf = wine_vec.transform([clean(query)])
    q_svd = svd_model.transform(q_tfidf)[0]
    terms = wine_vec.get_feature_names_out()

    def dim_terms(dim_idx):
        comp = svd_model.components_[dim_idx]
        return [terms[i] for i in comp.argsort()[-5:][::-1]]

    top_pos = q_svd.argsort()[-top_n:][::-1]
    top_neg = q_svd.argsort()[:top_n]

    return {
        'positive': [{'dim': int(d), 'value': float(q_svd[d]),
                      'terms': dim_terms(d)} for d in top_pos],
        'negative': [{'dim': int(d), 'value': float(q_svd[d]),
                      'terms': dim_terms(d)} for d in top_neg]
    }
