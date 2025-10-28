# 🎬 Movie Recommendation Chatbot

## 1. Introduction

### Context
With the rapid growth of audiovisual content, choosing a movie has become increasingly difficult. Streaming platforms offer thousands of options, making it hard for users to discover films that match their tastes. AI-powered chatbots leveraging natural language processing (NLP) provide an innovative solution to personalize user experiences.

### Objective
This project develops a chatbot capable of providing movie recommendations based on user preferences. By analyzing movie features such as genres, actors, directors, and keywords, the chatbot suggests relevant films tailored to individual tastes.

### Scope
- Content-based recommendation system using structured data (genres, descriptions, ratings).  
- Recommendations limited to a predefined movie dataset.  
- Features include search by title or actor.  
- Excludes social or real-time behavioral recommendations.

---

## 2. Dataset

The project uses an enriched version of the **Full MovieLens Dataset**, combined with **TMDb API** data.  
It contains metadata for ~45,000 movies and 26 million ratings from 270,000 users (up to July 2017).  

### Key Files
- `movies_metadata.csv` – Movie metadata (budget, revenue, release date, languages, production companies, posters, backdrops).  
- `keywords.csv` – Movie plot keywords (JSON objects).  
- `credits.csv` – Detailed cast and crew information (actors, directors).  
- `links.csv` – Links between TMDb and IMDb IDs.  
- `ratings_small.csv` – Subset of 100,000 ratings for 9,000 movies.  

> **Note:** Users must download the dataset from Kaggle and place the CSV files in a `data/` folder.  
> [Kaggle TMDb Movie Metadata Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 3. Model Description

### 3.1 Content-Based Recommendation
- Uses **Sentence-BERT (`all-mpnet-base-v2`)** to generate embeddings for movie descriptions.  
- User queries are also embedded to compute **cosine similarity** with movie embeddings.

### 3.2 Sentiment Analysis
- User preferences are analyzed to identify **positive** and **negative** tastes.  
- Negative preferences are used to filter out movies that the user dislikes.

### 3.3 Recommendation Pipeline
1. **Data preprocessing:** Load and merge `movies_metadata.csv`, `keywords.csv`, and `credits.csv`.  
2. **Combine fields:** Merge genres, keywords, cast, crew, ratings, and overview into a single `combine` column.  
3. **Embedding generation:** Create embeddings for movies and user queries.  
4. **Similarity computation:** Calculate cosine similarity between user query and movies.  
5. **Negative filtering:** Exclude movies that match negative user preferences.  
6. **Top-N selection:** Return the top N movies ranked by similarity.

---

## 4. Interface

The chatbot uses **Streamlit** for an interactive interface:

### Features
- **Chat input:** Users type preferences in natural language.  
- **Dynamic results:** Recommended movies appear in real-time.  
- **Information displayed:** Movie title, genre, main actors, brief description.  
- **Interactivity:** Results update automatically when preferences change.

### Example Queries
- `"I want a science-fiction adventure with female lead actors."`  
- `"No horror movies, only comedy or drama."`

---

## 5. Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/movie-chatbot.git
cd movie-chatbot
