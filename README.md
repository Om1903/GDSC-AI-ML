🎬 Movie Recommendation System
This project implements a Movie Recommendation System using Collaborative Filtering and Content-Based Filtering. It helps users discover movies based on their interests and preferences.

📌 Features
Collaborative Filtering: Suggests movies based on user behavior and ratings.
Content-Based Filtering: Recommends movies based on their metadata (genres, cast, crew, etc.).
Deep Learning: Uses PyTorch for training a neural network on movie embeddings.
Similarity Measures: Uses Cosine Similarity and KNN for finding similar movies.
📂 Dataset
We use IMDb Movies Dataset containing:

movies_metadata.csv: Movie details like ID, title, genres, cast, etc.
ratings.csv: User ratings for movies.
🚀 Installation & Setup
1️⃣ Install Required Libraries

pip install pandas numpy torch scikit-learn
2️⃣ Clone the Repository

git clone https://github.com/your-username/movie-recommendation.git
cd movie-recommendation
3️⃣ Run the Jupyter Notebook
Launch Jupyter Notebook and open Movie_Recommendation.ipynb.

🏆 Movie Recommendation Approaches
1️⃣ Collaborative Filtering (User-Item Interaction)
Collaborative Filtering suggests movies based on user interactions and ratings. We train a Neural Network using PyTorch on user-movie ratings.

🔹 Steps Involved:
Encode userId and movieId to continuous indices.
Train a neural network with embedding layers for users and movies.
Predict ratings for unseen movies.
Recommend movies based on predicted ratings.
🔹 Code Snippet:
python
Copy
Edit
# Neural Network for Collaborative Filtering
class Recommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super(Recommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)  

    def forward(self, user, movie):
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(movie)
        x = torch.cat([user_emb, movie_emb], dim=1)
        return self.fc(x).view(-1)
🔹 Get Movie Recommendations
python
Copy
Edit
def recommend_movies(user_id, movies_df, top_n=10):
    user_tensor = torch.tensor([user_id] * num_movies, dtype=torch.long)
    all_movie_ids = torch.arange(num_movies, dtype=torch.long)
    
    with torch.no_grad():
        predicted_ratings = model(user_tensor, all_movie_ids)
    
    top_movies = torch.argsort(predicted_ratings, descending=True)[:top_n]
    recommended_movie_titles = movies_df.iloc[top_movies.numpy()]['title'].values
    return recommended_movie_titles
2️⃣ Content-Based Filtering (Movie Metadata)
Content-Based Filtering recommends movies based on movie attributes like genres, cast, crew, and keywords. It uses Deep Learning and Cosine Similarity to find similar movies.

🔹 Steps Involved:
Convert movie genres, cast, and crew into numerical encodings.
Train a Neural Network to learn movie embeddings.
Compute Cosine Similarity to find similar movies.
🔹 Code Snippet:
python
Copy
Edit
# Neural Network for Movie Embeddings
class MovieEmbeddingModel(nn.Module):
    def __init__(self, num_movies, num_genres, num_cast, num_crew, num_keywords, embedding_dim=50):
        super(MovieEmbeddingModel, self).__init__()
        self.movie_embed = nn.Embedding(num_movies, embedding_dim)
        self.genre_embed = nn.Embedding(num_genres, embedding_dim)
        self.cast_embed = nn.Embedding(num_cast, embedding_dim)
        self.crew_embed = nn.Embedding(num_crew, embedding_dim)
        self.keyword_embed = nn.Embedding(num_keywords, embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  
        )

    def forward(self, x):
        movie_vec = self.movie_embed(x[:, 0])
        genre_vec = self.genre_embed(x[:, 1])
        cast_vec = self.cast_embed(x[:, 2])
        crew_vec = self.crew_embed(x[:, 3])
        keyword_vec = self.keyword_embed(x[:, 4])

        concat_vec = torch.cat((movie_vec, genre_vec, cast_vec, crew_vec, keyword_vec), dim=1)
        return self.fc(concat_vec)
🔹 Find Similar Movies Using KNN
python
Copy
Edit
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
knn.fit(movie_embeddings)

def recommend(movie_name, df, movie_embeddings, knn):
    index = df[df['title'] == movie_name].index[0]
    distances, indices = knn.kneighbors([movie_embeddings[index]], n_neighbors=11)
    
    recommendations = [df.iloc[i].title for i in indices[0][1:]]
    return recommendations

🔥 Results
The system provides personalized recommendations using user preferences and movie similarities.

✅ Collaborative Filtering:
Recommends movies based on user behavior.
🔹 Example: A user who liked Inception may get Interstellar recommended.

✅ Content-Based Filtering:
Finds movies similar to a given movie.
🔹 Example: If you like Inception, similar movies could be The Prestige, Memento, or Shutter Island.


💡 Improvements & Future Work
✅ Implement hybrid recommendation (collaborative + content-based).
✅ Use autoencoders for feature embeddings.
✅ Deploy as a web application using Flask/Django.


📜 Conclusion
Collaborative Filtering is user-dependent and requires interaction data.
Content-Based Filtering is movie-dependent and works even for new users.
Hybrid Models combining both can further improve recommendations.




