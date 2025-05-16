import streamlit as st
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import plotly.io as pio
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from utils import topic_diversity

st.set_page_config(layout="wide")
st.title("ER Analyzer : BERTopic on Elden Ring Textual Content")

container = st.container(border=True)

container.write("Welcome to ER Analyzer, the web app that allows you play with the hyperparameters of a BERTopic model and see how they impact topic representation!")
container.link_button("Click here to acces the GitHub page of this projet A MODIFIER", "www.google.com")

st.divider()

# --- Sidebar for parameters ---
st.sidebar.header("Model Parameters")
n_neighbors = st.sidebar.slider("UMAP - n_neighbors", 2, 50, 10)
n_components = st.sidebar.slider("UMAP - n_components", 2, 10, 5)
min_dist = st.sidebar.slider("UMAP - min_dist", 0.0, 1.0, 0.0)
umap_metric = st.sidebar.selectbox("UMAP - metric", ["cosine", "euclidean"])

min_cluster_size = st.sidebar.slider("HDBSCAN - min_cluster_size", 2, 50, 15)
vectorizer_min_df = st.sidebar.slider("CountVectorizer - min_df", 1, 10, 3)
top_n_words = st.sidebar.slider("BERTopic - top_n_words", 5, 30, 10)
nr_topics = st.sidebar.slider("BERTopic - nr_topics", 5, 100, 75)
ngram_range = st.sidebar.selectbox("N-gram Range", [(1,1), (1,2), (1,3)], index=1)
calculate_probabilities = st.sidebar.checkbox("Calculate Probabilities", value=False)

nbr_runs = st.sidebar.slider("Number of model trained", 1, 3, 5)

run_button = st.sidebar.button("Run")

if run_button:

# --- Load Data ---
    with st.status("Loading text data...", expanded=True):
        with open('data/data.json') as f:
            data = json.load(f)
            text_list = list(set([item for sublist in data.values() for item in sublist]))
        st.success("Data loaded successfully!")

    # --- Embeddings ---
    with st.status("Encoding with SentenceTransformer...", expanded=True):
        embedding_model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = embedding_model.encode(text_list, show_progress_bar=True)
        st.success("Embeddings generated.")

    # --- Model Setup ---
    with st.status("Configuring and training BERTopic model...", expanded=True):
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric=umap_metric)
        hdbscan_model = HDBSCAN(metric='euclidean', cluster_selection_method='eom', prediction_data=True, min_cluster_size=min_cluster_size)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=vectorizer_min_df)
        ctfidf_model = ClassTfidfTransformer()
        # representation_model = MaximalMarginalRelevance(diversity=0.3)

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            # representation_model=representation_model,
            top_n_words=top_n_words,
            nr_topics=nr_topics,
            n_gram_range=ngram_range,
            calculate_probabilities=calculate_probabilities,
            verbose=True
        )

    all_embeddings = []
    all_models = []
    for run_i in range(nbr_runs):

        with st.status(f"Training Model {run_i}", expanded=True):

            topics, probs = topic_model.fit_transform(text_list, embeddings)

            all_models.append(topic_model)

            all_embeddings.append(topic_model.topic_embeddings_)

            st.success(f"Run {run_i} - Topic Diversity : {topic_diversity(topic_model)}")

    n_models = len(all_embeddings)

    figs = []

    for i,j in combinations(range(n_models), 2):

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(all_embeddings[i], all_embeddings[j])

        # Plot the similarity matrix as a heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarity_matrix, fmt=".2f", cmap="viridis", 
                    xticklabels=[f"T2_{i}" for i in range(similarity_matrix.shape[1])],
                    yticklabels=[f"T1_{i}" for i in range(similarity_matrix.shape[0])])
        ax.set_title("Topic Similarity Between BERTopic Runs")
        ax.set_xlabel(f"Topics - Run {j}")
        ax.set_ylabel(f"Topics - Run {i}")
        plt.tight_layout()
        # plt.show()
        figs.append(fig)

        # Find indices of pairs with similarity > 0.8
        threshold = 0.8
        pairs = np.argwhere(similarity_matrix > threshold)

        # Extract values
        high_sim_pairs = [(i, j, similarity_matrix[i, j]) for i, j in pairs]
        st.write(f"Pourcentage topic similarity > 0.8: {len(high_sim_pairs)/75}")

    new_model = BERTopic.merge_models(all_models, min_similarity = 0.9, embedding_model=embedding_model)

    # to reassign topics and docs
    topics, probs = new_model.transform(text_list)

    new_model.update_topics(text_list, topics=topics, ctfidf_model=ctfidf_model, vectorizer_model=vectorizer_model)

    # --- Visualizations ---
    st.subheader("📊 Visualizations")

    tab1, tab2, tab3 = st.tabs(["📌 Topic Overview", "🌳 Hierarchy", "Similarity Matrixes"])

    with tab1:
        fig_topics = new_model.visualize_topics()
        st.plotly_chart(fig_topics, use_container_width=True)

    with tab2:
        fig_hierarchy = new_model.visualize_hierarchy()
        st.plotly_chart(fig_hierarchy, use_container_width=True)

    with tab3:
        for fig in figs:
            st.pyplot(fig)

    # --- Topic Info Table ---
    st.subheader("🧠 Top Topics")
    df_topics = new_model.get_topic_info()
    st.dataframe(df_topics)

    # Optional: Save model or CSV
    # topic_model.save("bertopic_model")
    # df_topics.to_csv("topics.csv", index=False)
