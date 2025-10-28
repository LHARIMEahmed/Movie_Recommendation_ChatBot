import pandas as pd
import numpy as np
import ast
import re
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import nltk
import torch
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

import openai
import streamlit as st
import shelve
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
import os
import json
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Load environment variables
load_dotenv()

# Set up Streamlit
st.title("Movie Recommendation Chatbot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load models and data
st.session_state.setdefault("embedding_model", SentenceTransformer('all-mpnet-base-v2'))
flan_t5_model_name = "google/flan-t5-small"
st.session_state.setdefault("flan_t5_tokenizer", AutoTokenizer.from_pretrained(flan_t5_model_name))
st.session_state.setdefault("flan_t5_model", AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_name))
flan_t5_model_name = "google/flan-t5-small"  # Example: Flan-T5 small model
tokenizer_flan_t5 = AutoTokenizer.from_pretrained(flan_t5_model_name)
model_flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_name)

# Sentence-BERT model for similarity computation
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Load GPT-2 for generating responses

@st.cache_data
def load_movie_data():
    movies_metadata = pd.read_csv("C:/Users/elelctro fatal/OneDrive/Bureau/movies_metadata.csv", low_memory=False)
    keywords = pd.read_csv("C:/Users/elelctro fatal/OneDrive/Bureau/keywords.csv", low_memory=False)
    credits = pd.read_csv("C:/Users/elelctro fatal/OneDrive/Bureau/credits.csv", low_memory=False)

    # Preprocessing data
    movies_metadata_df = pd.DataFrame(movies_metadata)
    keywords_df = pd.DataFrame(keywords)
    credits_df = pd.DataFrame(credits)

    # Process columns for genres, keywords, and credits
    movies_metadata_df['genres'] = movies_metadata_df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    movies_metadata_df['genres'] = movies_metadata_df['genres'].apply(lambda x: ' '.join([genre['name'] for genre in x]))
    keywords_df["keywords"] = keywords_df["keywords"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    credits_df["cast"] = credits_df["cast"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    credits_df["crew"] = credits_df["crew"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    credits_df['cast'] = credits_df['cast'].apply(lambda x: ' '.join([actor['name'] for actor in x]))
    credits_df['crew'] = credits_df['crew'].apply(lambda x: ' '.join([crew['name'] for crew in x if crew['job'] == 'Director']))

    # Handle missing values and combine text columns
    movies_metadata_df['tagline'] = movies_metadata_df['tagline'].fillna('')
    movies_metadata_df['overview'] = movies_metadata_df['overview'].fillna('')
    movies_metadata_df['description'] = movies_metadata_df['tagline'] + ' ' + movies_metadata_df['overview']

    # Convert IDs to numeric
    movies_metadata_df['id'] = pd.to_numeric(movies_metadata_df['id'], errors='coerce').astype('Int64')
    keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce').astype('Int64')
    credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce').astype('Int64')

    # Merge dataframes
    merged = pd.merge(movies_metadata_df, keywords_df, on='id')
    movies = pd.merge(merged, credits_df, on='id')

    # Clean and preprocess columns
    columns_to_combine = ['genres', 'keywords', 'cast', 'crew', 'vote_average', 'vote_count', 'overview']
    for car in columns_to_combine:
        movies[car] = movies[car].fillna('')

    movies[columns_to_combine] = movies[columns_to_combine].astype(str)

    # Combine relevant columns into one
    movies['combine'] = movies[columns_to_combine].agg(' '.join, axis=1)

    return movies

movies = load_movie_data()

# Extract preferences function
from transformers import pipeline
import re

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")

# Ensure the stopwords resource is downloaded
nltk.download('stopwords')
def extract_preferences_with_context(query):

    # Tokenize and preprocess the input query into meaningful segments
    segments = re.split(r'\b(and|but|or)\b', query, flags=re.IGNORECASE)
    
    positive_keywords = []
    negative_keywords = []

    # Analyze each segment for sentiment
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # Check the sentiment of the segment
        sentiment = sentiment_analyzer(segment)[0]

        if sentiment['label'] == 'POSITIVE':
            # Include the word 'love' only if it's used in the right context
            if "i love" in segment.lower():
                # Exclude 'love' in this case (if the context is 'love movie')
                segment = re.sub(r'\blove\b', '', segment, flags=re.IGNORECASE)
            positive_keywords.extend(re.findall(r'\b\w+\b', segment.lower()))
        
        elif sentiment['label'] == 'NEGATIVE':
            negative_keywords.extend(re.findall(r'\b\w+\b', segment.lower()))

    # Remove common stopwords (optional)
    stopwords_set = set(stopwords.words('english'))
    
    positive_keywords = [word for word in positive_keywords if word not in stopwords_set]
    negative_keywords = [word for word in negative_keywords if word not in stopwords_set]

    return positive_keywords, negative_keywords


# Recommendation function
def recommend_movies(movies, query, num_recommendations=5):
    # Extract positive and negative preferences
    positive_keywords, negative_keywords = extract_preferences_with_context(query)
    print(positive_keywords, negative_keywords)

    recommendations = movies  # Default to the original movie list

    # If there are positive keywords, proceed with generating recommendations
    if positive_keywords:
        # Generate embedding for the query based on positive preferences
        query_embedding = embedding_model.encode(' '.join(positive_keywords))

        # Generate embeddings for movie descriptions
        movie_descriptions = movies['combine'].tolist()
        movie_embeddings = embedding_model.encode(movie_descriptions)

        # Compute cosine similarities between query and movie descriptions
        cosine_similarities = cosine_similarity([query_embedding], movie_embeddings).flatten()

        # Add similarity scores to the DataFrame
        movies['similarity'] = cosine_similarities

        # If there are negative keywords, calculate their similarity with the movies
        if negative_keywords:
            # Generate embeddings for the negative preferences
            negative_keywords = negative_keywords[1:]
            negative_query = ' '.join(negative_keywords)
            negative_embedding = embedding_model.encode(negative_query)

            # Compute cosine similarity between the negative query and movie descriptions
            negative_similarities = cosine_similarity([negative_embedding], movie_embeddings).flatten()

            # Add the negative similarity to the DataFrame
            movies['negative_similarity'] = negative_similarities

            # Apply threshold to filter out movies with negative preferences
            threshold = 0.5  # Customize this threshold based on your needs
            movies = movies[movies['negative_similarity'] < threshold]

        # Rank movies based on positive preference similarity
        recommendations = movies.sort_values(by='similarity', ascending=False).head(num_recommendations)

    # If there are no positive preferences, handle negative keywords filtering
    elif negative_keywords:
        # Ignore the first element of the negative list (typically "hate," "dislike")
        filtered_negative_keywords = negative_keywords[1:]
        negative_query = ' '.join(filtered_negative_keywords)
        negative_embedding = embedding_model.encode(negative_query)

        # Generate embeddings for movie descriptions
        movie_descriptions = movies['combine'].tolist()
        movie_embeddings = embedding_model.encode(movie_descriptions)

        # Compute cosine similarity between the negative query and movie descriptions
        negative_similarities = cosine_similarity([negative_embedding], movie_embeddings).flatten()

        # Add the negative similarity to the DataFrame
        movies['negative_similarity'] = negative_similarities

        # Filter movies based on the negative similarity threshold
        threshold = 0.5
        filtered_movies = movies[movies['negative_similarity'] < threshold]

        # Randomly select a number of recommendations from the filtered list
        recommendations = filtered_movies.sample(n=min(num_recommendations, len(filtered_movies)))

    else:
        # If there are no preferences at all, return an empty message
        recommendations = pd.DataFrame()

    # If no recommendations are found, return a message
    if recommendations.empty:
        return "Sorry, no movies matched your preferences."

    # Prepare the final list of recommendations with title and genre in parentheses
    result = []
    for _, row in recommendations.iterrows():
        title = row['title']
        genre = row['genres']  # Assuming genres are a string like "Action Drama"
        result.append(f"{title} ({genre})")

    return "\n".join(result)



# Chat history and interface
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []

# Chat input and output
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# Chat input and output
greeting_keywords = ["hi", "hey", "hello", "start"]
farewell_keywords = ["bye", "exit", "goodbye", "see you", "quit"]

if prompt := st.chat_input("Tell me your movie preferences:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        try:
            # Check if the input contains any greeting keywords
            if any(greeting in prompt.lower() for greeting in greeting_keywords):
                response = "Hello! How can I help you with movie recommendations today?"
            # Check if the input contains any farewell keywords
            elif any(farewell in prompt.lower() for farewell in farewell_keywords):
                response = "Goodbye! It was a pleasure helping you. Come back anytime!"
            else:
                recommendations = recommend_movies(movies.head(200), prompt)
                
                # Check if recommendations is a DataFrame
                if isinstance(recommendations, pd.DataFrame):
                    response = recommendations.to_markdown(index=False)  # Convert to markdown if DataFrame
                else:
                    response = recommendations  # Use raw string message if not a DataFrame
        except Exception as e:
            response = f"An error occurred: {str(e)}"
        
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
