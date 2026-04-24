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
            'foods': foods[['idx', 'name', 'description', 'score']]
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

@app.route('/explain_food', methods=['POST'])
def explain_food():
    data = request.get_json()
    ingredients = data.get('ingredients', [])
    food_idx = int(data.get('food_idx'))

    from search import get_food_explanation
    explanation = get_food_explanation(ingredients, food_idx)
    return jsonify({'overlap': explanation})

@app.route('/rag', methods=['POST'])
def rag():
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("SPARK_API_KEY"),
        base_url="https://api.cerebras.ai/v1"
    )

    data = request.get_json()
    query       = data.get('query', '')
    wines       = data.get('wines', [])
    foods       = data.get('foods', [])
    flavors     = data.get('flavors', [])
    ingredients = data.get('ingredients', [])

    wine_context = '\n'.join([
        f"- {w.get('title','Unknown')} ({w.get('variety','')}): "
        f"{str(w.get('description',''))[:150]}"
        for w in wines[:3]
    ])

    food_context = '\n'.join([
        f"- {f.get('name','Unknown dish')}"
        for f in foods[:5]
    ])

    prompt = f"""A user is drinking a wine they describe as: "{query}"

Our system matched this description to wines like:
{wine_context}

Detected flavor profile: {', '.join(flavors)}
Complementary ingredients identified: {', '.join(ingredients[:8])}

Based on this, we are recommending these dishes:
{food_context}

Please write 3-4 sentences that:
1. Briefly characterize the wine based on the user's description
2. Explain why these dishes pair well with this wine style
3. Highlight the key flavor connections between the wine and food
4. IMPORTANT: For each dish, provide exactly one sentence describing what it is and why it works, formatted strictly as:
- **[Dish Name]**: [description]

Keep the tone conversational, like a knowledgeable friend explaining over dinner."""

    try:
        response = client.chat.completions.create(
            model="llama3.3-70b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        explanation = response.choices[0].message.content
        return jsonify({'explanation': explanation})
    except Exception as e:
        import traceback
        print(f"Error in /rag: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)