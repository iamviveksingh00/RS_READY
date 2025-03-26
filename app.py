import pickle

import requests
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load Data
movie_embeddings = pickle.load(open('movies_embeddings.pkl', 'rb'))
new = pickle.load(open('movie_list.pkl', 'rb'))

# Initialize NLP Model
model_new = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# RapidAPI Credentials
RAPIDAPI_KEY = '91bc8f445dmsh0bdf292d75b8c58p15bdf4jsn87b9450c30d1'
API_HOST = 'imdb8.p.rapidapi.com'


def fetch_poster(movie_name):
    """Fetch movie poster from IMDb API."""
    headers = {'X-RapidAPI-Key': RAPIDAPI_KEY, 'X-RapidAPI-Host': API_HOST}
    search_url = f"https://imdb8.p.rapidapi.com/title/find?q={movie_name}"

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'results' in data and len(data['results']) > 0:
            for result in data['results']:
                if 'image' in result and 'url' in result['image']:
                    return result['image']['url']
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    except requests.RequestException:
        return "https://via.placeholder.com/500x750?text=Error+Fetching+Poster"


def recommend_movies(user_input, method, top_n=5):
    """Return recommended movies based on user input and method."""
    if method == "MOVIE BASED":
        if user_input in new['title'].values:
            index = new[new['title'] == user_input].index[0]
            distances = sorted(list(enumerate(movie_embeddings[index])), reverse=True, key=lambda x: x[1])
            recommended_movies = new.iloc[[i[0] for i in distances[1:top_n + 1]]]
        else:
            return []
    
    elif method == "PERSONALIZED":
        query_embedding = model_new.encode([user_input])
        similarities = cosine_similarity(query_embedding, movie_embeddings)
        top_indices = similarities[0].argsort()[-top_n:][::-1]
        recommended_movies = new.iloc[top_indices]

    return [
        {"title": row['title'], "poster": fetch_poster(row['title'])} 
        for _, row in recommended_movies.iterrows()
    ]


@app.route('/')
def home():
    return render_template('index.html', movies=new['title'].tolist())


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint to get movie recommendations."""
    data = request.json
    user_input = data.get('user_input', '')
    method = data.get('method', 'MOVIE BASED')

    recommendations = recommend_movies(user_input, method)

    return jsonify({"movies": recommendations})


if __name__ == '__main__':
    app.run(debug=True)
