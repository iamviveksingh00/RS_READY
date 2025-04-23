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


def fetch_movie_details(movie_name):
    """Fetch movie details (poster, genre, cast, rating, overview) from IMDb API."""
    headers = {'X-RapidAPI-Key': RAPIDAPI_KEY, 'X-RapidAPI-Host': API_HOST}
    search_url = f"https://imdb8.p.rapidapi.com/title/find?q={movie_name}"

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'results' in data and len(data['results']) > 0:
            for result in data['results']:
                if 'id' in result:
                    movie_id = result['id']
                    details_url = f"https://imdb8.p.rapidapi.com/title/get-overview-details?tconst={movie_id}"
                    details_response = requests.get(details_url, headers=headers, timeout=10)
                    details_response.raise_for_status()
                    details_data = details_response.json()

                    # Extract required details
                    movie_details = {
                        "poster": result['image']['url'] if 'image' in result else "https://via.placeholder.com/500x750?text=No+Poster+Available",
                        "genre": details_data.get('genres', []),
                        "cast": details_data.get('crew', {}).get('cast', []),
                        "rating": details_data.get('rating', 'N/A'),
                        "overview": details_data.get('overview', 'No overview available')
                    }
                    return movie_details

        return {
            "poster": "https://via.placeholder.com/500x750?text=No+Poster+Available",
            "genre": [],
            "cast": [],
            "rating": "N/A",
            "overview": "No overview available"
        }
    except requests.RequestException:
        return {
            "poster": "https://via.placeholder.com/500x750?text=Error+Fetching+Poster",
            "genre": [],
            "cast": [],
            "rating": "N/A",
            "overview": "No overview available"
        }


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
        {
            "title": row['title'],
            "poster": fetch_movie_details(row['title'])['poster'],
            "genre": fetch_movie_details(row['title'])['genre'],
            "cast": fetch_movie_details(row['title'])['cast'],
            "rating": fetch_movie_details(row['title'])['rating'],
            "overview": fetch_movie_details(row['title'])['overview']
        } 
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