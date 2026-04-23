from flask import Flask, render_template, request, jsonify
from search import (search_wines, extract_flavors,
                    get_complementary_ingredients,
                    search_foods)
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        use_svd = data.get('use_svd', True)

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Step 1: Search for matched wine profiles
        wines = search_wines(query, use_svd=use_svd)
        
        # Step 2: Bridge it via FlavorDB concepts
        flavors = extract_flavors(wines)
        ingredients = get_complementary_ingredients(flavors)
        
        # Step 3: Match actual food recipes
        foods = search_foods(ingredients)
        
        return jsonify({
            'wines': wines[['idx', 'title', 'variety', 'description', 'score']]
                          .head(5).to_dict('records'),
            'flavors': flavors,
            'ingredients': ingredients,
            'foods': foods[['name', 'description', 'score']]
                          .head(8).to_dict('records'),
            'svd': use_svd
        })
    except Exception as e:
        print(f"Error in /search: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json()
    query = data.get('query')
    wine_idx = int(data.get('wine_idx'))

    from search import get_result_explanation
    explanation = get_result_explanation(query, wine_idx)
    return jsonify(explanation)

if __name__ == '__main__':
    app.run(debug=True, port=8000)