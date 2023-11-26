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

# function to clean the code tokens. Super rudimentary, 
# as of right now, we're just taking rid of the single punctuation
def clean_code_tokens(lst):
    result = string.punctuation 
    new_lst = [] 
    for character in lst:
        if character in result:
            continue
        else:
            new_lst.append(character)
    return new_lst

#From Hugging Face Tutorials
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = trained_model(**encoded_input)
    return cls_pooling(model_output)

#Function for cosine_similarity. #Look into np.cos Annoy FAISS. look into applying and vectorizing
def cosine_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


# Testing if the pytorch GPU functions work
print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available()) #We have GPU on deck and ready
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


# device = torch.device("cuda")
device = ("cuda" if torch.cuda.is_available() else "cpu")
hg_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_ckpt = hg_model #Can/Should test different models
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
trained_model = AutoModel.from_pretrained(model_ckpt)
trained_model.to(device)


class searchmodel:

    def __init__(self, data, file_paths, load = False):
        self.data = data
        self.load = load
        self.file_paths = file_paths #{"inverted_index" : "path_to_inverted_index", "embeddings_dataset" : "path_to_embedded_dataset"}

        self.inverted_index = None
        self.embed_dataset = None
        if load: #if the load boolean is true, file_paths is a dictionary to load info
            with open(file_paths["inverted_index"], 'rb') as f:  # open a text file
                self.inverted_index = pickle.load(f) # serialize the list
                f.close()

            with open(file_paths["embeddings_dataset"], 'rb') as f:  # open a text file
                self.embed_dataset = pickle.load(f) # serialize the list
                f.close()
            
            self.tsed_DF = self.embed_dataset.to_pandas()
            self.tsed_DF["clean_code_tokens"] =  self.tsed_DF["func_code_tokens"].apply(clean_code_tokens)
        else:
            self.make_embeddings(self.data)
            self.make_inverted_index()
        self.create_tfidf()

    def make_inverted_index(self):
        tsed_DF = self.embed_dataset.to_pandas()

        
        # creating a column of "clean" code tokens
        # There's many many issues with this strategy
        tsed_DF["clean_code_tokens"] =  tsed_DF["func_code_tokens"].apply(clean_code_tokens)

        # Creates list of documents
        # documents = tsed_DF["clean_code_tokens"].to_dict()

        # Compiles a list of the words 
        all_words = []
        for i in list(tsed_DF["clean_code_tokens"].to_dict().values()):
            all_words += i

        #convert all words to a set, eliminates, duplicates
        all_words = list(set(all_words)) #Get rid of all repeats
        # all_words

        inverted_index = {}
        for i in range(len(tsed_DF)):
            token_counter = Counter(tsed_DF.iloc[i]["clean_code_tokens"])

            for token in token_counter:
                if token not in inverted_index:
                    inverted_index[token] = {}
                inverted_index[token][i] = token_counter[token]
        
        #Pickle afterwards
        with open(self.file_paths["inverted_index"], 'wb') as f:  # open a text file
            pickle.dump(inverted_index, f) # serialize the list
            f.close()
        
        self.inverted_index = inverted_index
        self.tsed_DF = tsed_DF


    def create_tfidf(self):
        num_rows = len(self.tsed_DF)

        tf_idf = {}
        for i in range(num_rows):
        # print(i)
            tokens = self.tsed_DF["clean_code_tokens"].iloc[i]
            counter = Counter(tokens)
            words_count = len(tokens)

            for token in np.unique(tokens):
                tf = counter[token] / words_count
                df = len(self.inverted_index[token])
                idf = np.log((num_rows + 1) / (df + 1))

                tf_idf[i, token] = tf * idf
        self.tf_idf = tf_idf


    def make_embeddings(self, dataset):
        # model_ckpt = hg_model #Can/Should test different models
        # self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        # model = AutoModel.from_pretrained(model_ckpt)

        # Load the model to the GPU. Mine is a 3060
        # self.device = torch.device("cuda")
        # trained_model.to(device)

        dataset_embeddings = dataset.map(
            lambda x: {"embeddings": get_embeddings(x["func_documentation_string"]).detach().cpu().numpy()[0]}
        )

        dataset_embeddings.add_faiss_index(column="embeddings")

        with open(self.file_paths["embeddings_dataset"], 'wb') as f:  # open a text file
            pickle.dump(dataset_embeddings, f) # serialize the list
            f.close()

        self.embed_dataset = dataset_embeddings

    
    
    def query_results(self, query_string, k = 10):
        query_tokens = query_string.split()

        rel_indices = []
        
        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices += list(self.inverted_index[token].keys())
        
        rel_indices = set(rel_indices)

        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        # len(query_embedding[0])
        # len(tsed_DF["embeddings"][0])
        

        result_lst = []
        for i in rel_indices:
            for token in query_tokens:
                tf_score = 0
                try:
                    tf_score += (self.tf_idf[(i, token)])
                except: continue #this is bad, make sure this isn't the play
            # print(i)

            result_lst.append([i, tf_score, cosine_sim(self.tsed_DF["embeddings"][i], query_embedding[0])])
        
        result_lst.sort(reverse=True, key = lambda x: 0.5 * x[1] + 0.5*x[2])
        return result_lst[:k]


    # Function which runs all 99 queries, and returns a pd df of the results
    def create_results(self, query_filepath, results_per_query = 100):
        queries = pd.read_csv(query_filepath)
        # display(queries)
        q_lst = queries["query"].to_list()
        # print(q_lst)

        lang_lst = []
        func_code_url_lst = []
        query_lst = []

        for i, query in enumerate(q_lst):
            # print(i)
            fbm_lst = self.query_results(query, results_per_query)
            query_lst += [query for j in range(len(fbm_lst))]
            
            for lst in fbm_lst:
                # print(tsed_DF.iloc[lst[0]]["language"])
                # print(tsed_DF.iloc[lst[0]]["func_name"])
                # print(tsed_DF.iloc[lst[0]]["func_code_url"])
                # print(f"SCORE: {lst[1]}")
                # print("-" * 100)

                lang_lst.append(self.tsed_DF.iloc[lst[0]]["language"])
                func_code_url_lst.append(self.tsed_DF.iloc[lst[0]]["func_code_url"])
            
            # break

        # print(lang_lst)
        # print(func_code_url_lst)
        # print(query_lst)
        prediction_df = pd.DataFrame({'language' : lang_lst, 'url': func_code_url_lst, "query" : query_lst})
        return prediction_df