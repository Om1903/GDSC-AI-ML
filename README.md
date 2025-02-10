🎬 Movie Recommendation System (Collaborative Filtering)
This project implements a Collaborative Filtering-based Movie Recommendation System using PyTorch. The model is trained to predict user ratings for movies and recommend the top movies for a given user.

📌 Features
Uses PyTorch for deep learning-based collaborative filtering.
Implements user and movie embeddings to learn feature representations.
Predicts movie ratings for users based on historical data.
Provides personalized recommendations for each user.
Uses cosine similarity to improve recommendations.
📂 Dataset
The system is trained on the MovieLens dataset, which consists of user ratings for movies.
Required CSV files:

ratings.csv → Contains user ratings (userId, movieId, rating).
movies_metadata.csv → Contains movie details (id, title, genres).
📥 Download Dataset:
You can download the dataset from MovieLens or use a similar dataset.

🛠️ Prerequisites
Ensure you have the following installed before running the project:

🔹 Install Python Libraries
Run the following command in your terminal:

bash
Copy
Edit
pip install numpy pandas torch scikit-learn
🔹 Additional Dependencies
Jupyter Notebook (if running interactively):
bash
Copy
Edit
pip install notebook
Ensure PyTorch is installed. You can check by running:
bash
Copy
Edit
python -c "import torch; print(torch.__version__)"
🚀 Running the Project
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/movie-recommendation.git
cd movie-recommendation
2️⃣ Place the Dataset
Ensure ratings.csv and movies_metadata.csv are inside the project directory.
3️⃣ Run Jupyter Notebook
bash
Copy
Edit
jupyter notebook
Open collaborative_filtering.ipynb and run all cells.
4️⃣ Run the Python Script (Alternative)
If you prefer running as a script:

bash
Copy
Edit
python collaborative_filtering.py
📝 How to Use the Recommender
Once the model is trained, use the function below to get recommendations:

python
Copy
Edit
recommend_movies(user_id=1000, movie_df=movies_df, top_n=10)
Replace 1000 with any user ID.
The system will return the top 10 recommended movies for the user.
🎥 Example Output
markdown
Copy
Edit
Top 10 Recommended Movies for User 1000:
1. The Dark Knight
2. Inception
3. Interstellar
4. The Prestige
5. The Matrix
6. Memento
7. Fight Club
8. The Lord of the Rings: The Fellowship of the Ring
9. The Godfather
10. Pulp Fiction
🔍 Code Overview
Collaborative Filtering Model
Uses embedding layers for both users and movies.
Applies a fully connected layer to predict ratings.
Optimized using MSE Loss & Adam Optimizer.
Movie Recommendation Function
Predicts ratings for all movies a user hasn't watched.
Sorts the movies by predicted rating.
Returns the top N recommended movie titles.
💡 Improvements & Future Work
✅ Implement hybrid recommendation (collaborative + content-based).
✅ Use autoencoders for feature embeddings.
✅ Deploy as a web application using Flask/Django.
