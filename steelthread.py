import relevanceeval
import steelthread
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


dataset_dict = datasets.load_from_disk("./Dataset/CodeSearchCorpus/")
train_dataset = dataset_dict["train"]

num_rows = 5000
filepath_pkl_obj = "./PickleObjects/"
inverted_index_name = f"inverted_index_{num_rows}.pkl"
tsed_name = f"train_subset_embeddings_dataset_{num_rows}.pkl"

print(inverted_index_name, tsed_name)

np.random.seed(1)
train_subset_indices = np.random.choice(len(train_dataset), num_rows, replace = False)
train_dataset_subset = train_dataset.select(train_subset_indices)

print(len(train_dataset_subset))

file_paths = {"inverted_index" : f"./pickleObjects/{inverted_index_name}", "embeddings_dataset" : f"./pickleObjects/{tsed_name}"}
mod = searchmodel.searchmodel(train_dataset_subset, file_paths, load = True)

# print(mod.inverted_index)
# print(len(mod.embed_dataset))

#REMEMBER TO CHANGE TH CSV NAME
res_df = mod.create_results("./Dataset/Testing/queries.csv", results_per_query=50)
res_df.to_csv("./csv_output/baseline_5k.csv")

gen_results.create_lj_answers("./csv_output/baseline_5k.csv", "./Dataset/Testing/annotationStore.csv", "./csv_output/rel_5k.csv")