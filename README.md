# Academic Search Engine Using FAISS
This project is an academic search engine built using [Streamlit](https://streamlit.io/), [FAISS](https://github.com/facebookresearch/faiss), and [Sentence Transformers](https://www.sbert.net/). The engine allows users to search for academic papers by querying titles, authors, venues, publication years, and abstracts from a dataset. The search is powered by FAISS (Facebook AI Similarity Search) for efficient similarity-based lookups on embeddings generated using a pre-trained transformer model.

## Features
- **Search Academic Papers**: Users can input a search query, and the engine returns relevant academic papers by searching through the dataset.
- **Efficient Searching**: Uses FAISS to perform fast similarity-based searches.
- **Transformer Model**: Uses the `paraphrase-MiniLM-L12-v2` model to encode text into embeddings.

## Prerequisites
Ensure you have the following dependencies installed:
- Python 3.7+
- Streamlit
- FAISS
- Sentence Transformers
- Pandas
- NumPy
- Pickle

You can install the required packages using the following command:
- *pip install -r requirements.txt*

## Dataset
The dataset used in this project is the DBLP dataset, which contains metadata of academic papers such as titles, authors, venues, publication years, and abstracts. The dataset should be provided as a CSV file (dblp-v10.csv). It must be downloaded from here: *https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip*

## How It Works
- **Embeddings Generation**: The project uses the Sentence Transformers library to generate embeddings from the paper titles. These embeddings are then indexed using FAISS for similarity-based search.
- **FAISS Index**: The FAISS index allows fast searches based on the similarity of the query embeddings to the paper titles' embeddings.

## Usage
- Step 1: Generate the FAISS Index
Run the model.py script to generate embeddings and save the FAISS index:
*python model.py*
This will create two files:
*faiss_index.bin*: The FAISS index containing the paper titles' embeddings.
*titles.pkl*: A pickle file containing the list of paper titles.
- Step 2: Run the Streamlit App
Once the index and embeddings are saved, you can run the Streamlit app:
*streamlit run app.py*
This will launch a web interface where users can input search queries and receive results based on the similarity of their query to paper titles.

## Example Queries
Try searching for:
"Machine learning"
"Natural language processing"
"Deep learning"
The engine will return a list of academic papers that match the query.