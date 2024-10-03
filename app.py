# app_with_faiss.py
import time
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# Load and prepare data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path).dropna()
    return df['title'].values, df['authors'].values, df['venue'].values, df['year'].values, df['abstract'].values

# Load embeddings and FAISS index
@st.cache_resource
def load_faiss_index(embedding_file='faiss_index.bin', titles_file='titles.pkl'):
    index = faiss.read_index(embedding_file)
    with open(titles_file, 'rb') as f:
        titles = pickle.load(f)
    return index, titles

# Perform search
def search_titles(query, faiss_index, titles, authors, venue, years, abstracts, model, top_n=5):
    start = time.time()
    query_embedding = model.encode([query], show_progress_bar=True)
    query_embedding = np.array(query_embedding, dtype='float32')
    distances, indices = faiss_index.search(query_embedding, top_n)
    end = time.time()

    st.write(f'Search Results({end-start:.2f} seconds):')
    
    indices = indices.flatten()

    for idx in indices:
        if idx < len(titles):
            st.write(f"Title: {titles[idx]}")
            st.write(f"Authors: {authors[idx]}")
            st.write(f"Publication Date: {years[idx]}")
            st.write(f"Venue: {venue[idx]}")
            st.write(f"Abstract: {abstracts[idx]}")
            st.write("------------------------------------------------------")

# Streamlit UI
st.title('Academic Search Engine')

# Load data
file_path = 'dblp-v10.csv'
titles, authors, venue, years, abstracts = load_data(file_path)

# Load embeddings and FAISS index
faiss_index, titles_embeddings = load_faiss_index()

# Load the model (for query encoding)
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Input for search query
query = st.text_input('Enter a search query:', '')

if st.button('Search'):
    if query:
        search_titles(query, faiss_index, titles, authors, venue, years, abstracts, model, top_n=20)
    else:
        st.write('Please enter a query')
