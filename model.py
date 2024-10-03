# train_and_save_embeddings_with_faiss.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load and prepare data
def load_data(file_path):
    df = pd.read_csv(file_path).dropna()
    return df['title'].values

# Embed the data using the model
def embed_data(titles, model_name='paraphrase-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(titles.tolist(), show_progress_bar=True)
    return embeddings

# Main function to train and save embeddings with FAISS index
def main():
    file_path = 'dblp-v10.csv'
    titles = load_data(file_path)
    
    if titles is not None:
        embeddings = embed_data(titles)
        
        # Create FAISS index
        dimension = embeddings.shape[1]  # Dimension of embeddings
        index = faiss.IndexFlatL2(dimension)  # Flat L2 distance index
        index.add(embeddings)  # Add embeddings to index
        
        # Save embeddings and FAISS index
        faiss.write_index(index, 'faiss_index.bin')
        with open('titles.pkl', 'wb') as f:
            pickle.dump(titles, f)
        
        print("Embeddings and FAISS index saved successfully.")

if __name__ == "__main__":
    main()
