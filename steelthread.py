"""Steel Thread

This script runs the 99 queries and calculates the NDCG.

If first takes in the CodeSearchNet dataset, then takes a subset, trains a
search engine model, and predicts the preset 99 queries and calculates
the NDCG. 

Saves the results into the "csv_output/" folder.

Methods
----------
load_data:
    Loads in the dataset with the given number of rows and seed

create_model:
    Initialize the searchmodel instance with the given inputs

create_results:
    Runs the 99 queries and calculates the NDCG score.
"""
import gen_results
import searchmodel
import pandas as pd
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from datasets import load_dataset
import datasets
import torch
from collections import Counter
import string
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import faiss

def load_data(num_rows = 100000, seed = 1):
    """
    load_data function
    
    Loads a portion or all of the CodeSearchCorpus data 

    Parameters
    ----------
    num_rows: int
        The number of rows to sample
    
    seed:  int
        The random seed to set the dataset shuffling.
    """
    dataset_dict = datasets.load_from_disk("./Dataset/CodeSearchCorpus/")
    train_dataset = dataset_dict["train"]
    #obtain the number of rows, and name the inverted index and embeddings dataset
    
    # filepath_pkl_obj = "./PickleObjects/"
    inverted_index_name = f"inverted_index_{num_rows}_docs_bigrams.pkl"
    tsed_name = f"train_subset_embeddings_dataset_{num_rows}_cb.pkl"
    bigrams_set_name = f"bigrams_set_{num_rows}_docs_bigrams.pkl"
    bm_avg_DL_name = f"bm_avg_DL_{num_rows}.pkl"

    #Create the train data subset
    np.random.seed(seed)
    train_subset_indices = np.random.choice(len(train_dataset), num_rows, replace = False)
    train_dataset_subset = train_dataset.select(train_subset_indices)
    train_dataset_subset = train_dataset_subset.select_columns(["func_documentation_string", "language", "func_code_url", "whole_func_string", "func_name"])
    # print(len(train_dataset_subset))

    #Set the filepaths and pickle object names
    file_paths = {"inverted_index" : f"./pickleObjects/{inverted_index_name}", "bigrams_set" : f"./pickleObjects/{bigrams_set_name}", "bm_avg_DL" : f"./pickleObjects/{bm_avg_DL_name}",
                  "embeddings_dataset" : f"./pickleObjects/{tsed_name}"}

    return train_dataset_subset, file_paths
    


def create_model(num_rows, seed, loadEmbed, loadTF, query_function, kw_method, tf_alpha, bigrams):
    """
    create_model function
    
    Creates an instance of the searchmodel given the accompanying arguments 

    Parameters
    ----------
    num_rows: int
        The number of rows to sample
    
    seed:  int
        The random seed to set the dataset shuffling.
    
    loadEmbed: bool
        Boolean setting whether the embedding data needs to be made or loaded

    loadTF: bool
        Boolean setting whether the inverted index data needs to be made or loaded

    query_function: str
        String representing the specific function the model will use to generate results

    kw_method: str
        String, either "TFIDF" or "BM25" that sets the key term matching algorithm.

    tf_alpha: float
        Float value between [0, 1] that dictates the linear combination weights of the term frequency
        with the semantic similarity.

    bigrams: bool
        Boolean representing the use of bigrams or not in the inverted index and search algorithm

    """
    print(f"Creating model of {num_rows} rows, seed: {seed}, query method: {query_function} \n \
          {kw_method}, Bigram support {bigrams}, tf_alpha (if necessary): {tf_alpha}")
    train_dataset_subset, file_paths = load_data(num_rows, seed)

    mod = searchmodel.searchmodel(train_dataset_subset, file_paths, query_function, kw_method, tf_alpha, bigrams,loadEmbed, loadTF)

    print("Model done initializing")
    return mod

#Create the results of the 99 queries
def create_results(mod, num_rows):
    """
    create_results
    
    Function that takes the 99 queries and calculates the NDCG

    Parameters
    ----------
    mod: searchmodel 
        searchmodel instance
    
    num_rows: int
        Returns num_rows amount of the top results per query
    """
    print("Creating results now")
    res_df = mod.create_results("./Dataset/Testing/queries.csv", results_per_query=50)
    res_df.to_csv(f"./csv_output/baseline_{num_rows}k.csv")

    #Calculate the NDCG of the 99 queries
    gen_results.create_lj_answers_NEW(f"./csv_output/baseline_{num_rows}k.csv", "./Dataset/Testing/annotationStore_UNIQUE.csv")