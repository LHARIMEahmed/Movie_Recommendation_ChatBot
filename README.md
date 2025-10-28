Movie Recommendation Chatbot
1. Introduction
Context

With the ever-growing amount of audiovisual content, choosing a movie has become increasingly complex. Streaming platforms offer thousands of options, making it difficult for users to discover films tailored to their preferences. AI-powered chatbots leveraging natural language processing (NLP) offer innovative solutions to personalize user experiences.

Objective

This project develops a chatbot that provides movie recommendations based on user preferences. By analyzing movie features (genres, actors, directors, keywords), the chatbot suggests relevant movies aligned with the user’s tastes.

Scope

Content-based recommendation system using structured data (genres, descriptions, ratings).

Recommendations limited to a predefined movie dataset.

Features include search by title or actor.

Excludes social or real-time behavioral recommendations.

2. Dataset

The project uses an enriched version of the Full MovieLens Dataset combined with TMDb API data, containing metadata for ~45,000 movies and 26 million ratings from 270,000 users (up to July 2017).

Key Files

movies_metadata.csv – Main metadata including budget, revenue, release date, languages, production companies, posters, and backdrops.

keywords.csv – Movie plot keywords (JSON objects).

credits.csv – Detailed cast and crew information (actors, directors).

links.csv – Links between TMDb and IMDb IDs.

ratings_small.csv – Subset of 100,000 ratings for 9,000 movies.

Note: Users must download the dataset from Kaggle: TMDb Movie Metadata Dataset
 and place the CSV files in a data/ folder.

3. Model Description
3.1 Content-Based Recommendation

The chatbot uses Sentence-BERT (all-mpnet-base-v2) to compute semantic embeddings of movie descriptions. User queries are also transformed into embeddings, allowing the system to compute cosine similarity between user preferences and movie features.

3.2 Sentiment Analysis

User preferences are analyzed using sentiment analysis to distinguish between positive and negative tastes. Negative preferences are used to filter out undesirable movies from recommendations.

3.3 Recommendation Pipeline

Data preprocessing: Clean and merge movies_metadata.csv, keywords.csv, and credits.csv.

Combine relevant fields into a single combine column (genres, keywords, cast, crew, ratings, overview).

Embedding generation: Create embeddings for all movies and user queries.

Similarity computation: Compute cosine similarity between the query and movie embeddings.

Negative filtering: Remove movies matching negative user preferences.

Top-N selection: Return the top N movies ranked by similarity.

4. Interface

The chatbot uses Streamlit for an interactive interface:

Chat input: Users type their preferences in natural language.

Dynamic results: Recommended movies are displayed in real-time, ranked by relevance.

Information displayed: Movie title, genre, main actors, and a brief description.

Interactivity: Results update automatically when user preferences change.

5. Installation

Clone the repository

git clone https://github.com/yourusername/movie-chatbot.git
cd movie-chatbot


Install dependencies

pip install -r requirements.txt


Download dataset from Kaggle
Place movies_metadata.csv, keywords.csv, credits.csv in a data/ folder.
Kaggle TMDb Dataset

Run the chatbot

streamlit run app.py
