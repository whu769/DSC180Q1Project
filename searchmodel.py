"""searchmodel Class

This class trains a search model object

It either trains or loads pre-existing embedding models or inverted indices.
The model then saves newly made inverted indices and embedding models to 
speed up the process. 

The models has various querying methods from FAISS similarity scoring, to only TF-IDF

"""
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
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import RegexpStemmer
from nltk.util import ngrams
from nltk.corpus import stopwords

# CLASS FUNCTIONS BELOW

#From HuggingFace Tutorials
def cls_pooling(model_output):
    """
    Function that helps in creating the semantic search embeddings
    Code was taken from HuggingFace tutorials

    Parameters
    ----------
    model_output : str
        The file location of the answers csv file
    """
    return model_output.last_hidden_state[:, 0]

#From HuggingFace Tutorials
def get_embeddings(text_list):
    """
    Function that obtains the semantic search embeddings
    given a list of strings comprising of the tokens

    Parameters
    ----------
    text_list : list
        The list of string tokens to obtain the embeddings
    """
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = trained_model(**encoded_input)
    return cls_pooling(model_output)

#Function for cosine_similarity. 
def cosine_sim(a, b):
    """
    Class Function that calculates the cosine similarities of two vectors

    Parameters
    ----------
    a : list
        The list vectors to be compared 
    """
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


# Uncomment this portion to check for PyTorch GPU Availability
# print(torch.backends.cudnn.enabled)
# print(torch.cuda.is_available()) 
# print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


# Code to setup the pytorch functionality
device = ("cuda" if torch.cuda.is_available() else "cpu") #Chooses cuda if a cuda gpu is available
hg_model = "huggingface/CodeBERTa-small-v1" #"sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_ckpt = hg_model
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
trained_model = AutoModel.from_pretrained(model_ckpt)
trained_model.to(device) #moves the model to the proper device (cpu or cuda)

#Tokenizer initialization
st = RegexpStemmer('ing$|s$|e$|able$', min=4)
rgx_tokenizer = RegexpTokenizer(r'\w+')

class searchmodel:
    """
    searchmodel class

    Class Attributes
    ----------
    self.data: HuggingFace Datasets
        The CodeSearchNet data (from HuggingFace Dataset)
    
    self.loadEmbed: bool
        Boolean that states whether the embeddings data can be loaded or not
    
    self.loadTF: bool
        Boolean that states whether the inverted_index data can be loaded or not
    
    self.file_paths: dict
        A dict that has two keys: inverted_index and embeddings_dataset. It points to where pre-trained
        pickle objects exist OR where the trained objects should be pickled

    self.inverted_index: dict
        A dictionary that is the inverted index.

    self.embed_dataset: HuggingFace Datasets
        Dataset containing the semantic embeddings.

    Methods
    ----------
    make_inverted_index_docs:
        A function to make an inverted index without bigram support. (The first implementation and may be outdated)
    
    make_inverted_index_docs_bigrams:
        A function to make an inverted index with bigram support. Used as the primary inverted_index creation method
    
    make_inverted_index_docs_bigrams_only:
        An experimental function to make an inverted index comprising exclusively of bigrams. Was made as a test. Not 
        used normally.
    
    make_embeddings:
        Function to create the semantic embeddings based off the given pre-trained model. It will also be saved as a pickle
        file
    
    create_results:
        Function that takes in the 99 test queries and returns it as a Pandas DataFrame
    
    query_results_lc_naive_custom:
        A method of querying for results. It utilizes a linear combination of keyword matching and embedding cosine values.
        It is considered "naive" as it doesn't normalize and as so the keyword portion has more weight naturally.
        You can set the weights for the linear combination from 0-1.
    
    query_results_tfidf:
        Method of querying for results using only the tf-idf method from the inverted index
    
    query_results_BM25:
        Method of querying for results using only BM25 and the inverted index's information
    
    query_results_lc_norm:
        A method of querying for results using a linear combination of a keyword matching algorithm
        (either BM25 or TFIDF) and semantic embeddings cosine similarity. The keyword matching however 
        is normalized to 0-1 by dividing all scores by the max.
    
    query_results_odds_evens:
        A method of querying for the results using both keyword matching and semantic embeddings FAISS. The
        top results are then put in alternating order (semantic results are evens, keyword results are odds)
    
    query_results_faiss_kw:
        A method of querying results by first using FAISS on the semantic embeddings and then ranking 
        the returned results with a keyword matching algorithm.
    
    query_results_faiss_cos:
        A method of querying results using purely semantic embeddings by first getting the top nearest 
        functions with FAISS, then calculating the cosine similarity of those.
    """

    def __init__(self, data, file_paths, loadEmbed = False, loadTF = False):
        """ __init__ method
        Initializes a searchmodel object

        Parameters
        ----------
        data: HuggingFace Dataset
            The HuggingFace Dataset which may or may not be a subset of the overall CodeSearchNet data
    
        file_paths: dict
            A dict that has two keys: inverted_index and embeddings_dataset. It points to where pre-trained
            pickle objects exist OR where the trained objects should be pickled
        
        loadEmbed: bool
            A boolean that is True/False depending on whether an embedding model exists and doesn't need to 
            be trained
        
        loadTF: bool
            A boolean that is True/False depending on whether an inverted_index exists already. If False, 
            the model will train one from the data first
        """
        self.data = data
        self.loadEmbed = loadEmbed
        self.loadTF = loadTF
        self.file_paths = file_paths

        self.inverted_index = None
        self.embed_dataset = None

        #Load or make the inverted index and embed_dataset
        if loadEmbed:
            with open(file_paths["embeddings_dataset"], 'rb') as f:  # open a text file
                self.embed_dataset = pickle.load(f) # serialize the list
                f.close()
            
            self.tsed_DF = self.embed_dataset.to_pandas()
        else:
            self.make_embeddings(self.data)

        if loadTF: #if the load boolean is true, file_paths is a dictionary to load info
            with open(file_paths["inverted_index"], 'rb') as f:  # open a text file
                self.inverted_index = pickle.load(f) # serialize the list
                f.close()
        else:
            self.make_inverted_index_docs_bigrams()

    # No bigrams
    def make_inverted_index_docs(self):
        """
        make_inverted_index_docs

        Makes an inverted index based on the "func_documentation_string" of the given dataset
        """
        # tsed_DF = self.embed_dataset.select_columns(["func_documentation_string", "embeddings"])
        stop_words = set(stopwords.words('english'))
        tsed_DF = self.embed_dataset.to_pandas()
        tsed_DF["func_doc_tokens"] = tsed_DF["func_documentation_string"].apply(lambda x: rgx_tokenizer.tokenize(x))
        tsed_DF["func_doc_stem_tokens"] = tsed_DF["func_doc_tokens"].apply(lambda x: [st.stem(word) for word in x if word not in stop_words])
        inverted_index = {}
        for i in range(len(tsed_DF)):
            token_counter = Counter(tsed_DF.iloc[i]["func_doc_stem_tokens"])

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
    
    # With bigrams
    def make_inverted_index_docs_bigrams(self):
        """
        make_inverted_index_docs

        Makes an inverted index based on the "func_documentation_string" of the given dataset.
        There is bigram support for this.
        """

        stop_words = set(stopwords.words('english'))
        tsed_DF = self.embed_dataset.to_pandas()
        tsed_DF["func_doc_tokens"] = tsed_DF["func_documentation_string"].apply(lambda x: rgx_tokenizer.tokenize(x))
        tsed_DF["func_doc_stem_tokens"] = tsed_DF["func_doc_tokens"].apply(lambda x: [st.stem(word.lower()) for word in x if word not in stop_words])
        inverted_index = {}

        bigram_lst = []
        for i in range(len(tsed_DF)):
            bigram_lst += list(ngrams(tsed_DF.iloc[i]["func_doc_stem_tokens"], 2))
        
        bigram_counter = Counter(bigram_lst)

        # COME BACK TO THIS LATER
        # bigram_counter = (Counter({k: c for k, c in bigram_counter.items() if c <= 1000 and c >= 50}))

        self.bigram_set = set(list(bigram_counter.keys()))
        total_len_sum = 0

        for i in range(len(tsed_DF)):
            token_counter = Counter(tsed_DF.iloc[i]["func_doc_stem_tokens"])
            bigram_counter_i = Counter(list(ngrams(tsed_DF.iloc[i]["func_doc_stem_tokens"], 2)))
            total_len = sum(token_counter.values()) + sum(bigram_counter_i.values())
            total_len_sum += total_len

            for token in token_counter:
                if token not in inverted_index:
                    inverted_index[token] = {}
                inverted_index[token][i] = (token_counter[token], total_len)
            
            for bigram in bigram_counter_i:
                if bigram in self.bigram_set:
                    if bigram not in inverted_index:
                        inverted_index[bigram] = {}
                    inverted_index[bigram][i] = (bigram_counter_i[bigram], total_len)
        
        #Pickle afterwards
        with open(f"{self.file_paths['inverted_index']}", 'wb') as f:  # open a text file
            pickle.dump(inverted_index, f) # serialize the list
            f.close()
        
        self.bm_avg_DL = total_len_sum / len(tsed_DF)
        self.inverted_index = inverted_index
        self.tsed_DF = tsed_DF
    
    # Only bigrams
    def make_inverted_index_docs_bigrams_only(self):
        """
        make_inverted_index_docs_bigrams_only

        Makes an inverted index based on the "func_documentation_string" of the given dataset.
        It creates an inverted index comprising ONLY OF BIGRAMS
        """
        # tsed_DF = self.embed_dataset.select_columns(["func_documentation_string", "embeddings"])
        stop_words = set(stopwords.words('english'))
        tsed_DF = self.embed_dataset.to_pandas()
        tsed_DF["func_doc_tokens"] = tsed_DF["func_documentation_string"].apply(lambda x: rgx_tokenizer.tokenize(x))
        tsed_DF["func_doc_stem_tokens"] = tsed_DF["func_doc_tokens"].apply(lambda x: [st.stem(word) for word in x if word not in stop_words])
        inverted_index = {}

        bigram_lst = []
        for i in range(len(tsed_DF)):
            bigram_lst += list(ngrams(tsed_DF.iloc[i]["func_doc_stem_tokens"], 2))
        
        bigram_counter = Counter(bigram_lst)

        # COME BACK TO THIS LATER
        # bigram_counter = (Counter({k: c for k, c in bigram_counter.items() if c <= 1000 and c >= 50}))

        self.bigram_set = set(list(bigram_counter.keys()))

        for i in range(len(tsed_DF)):
            # token_counter = Counter(tsed_DF.iloc[i]["func_doc_stem_tokens"])
            bigram_counter_i = Counter(list(ngrams(tsed_DF.iloc[i]["func_doc_stem_tokens"], 2)))

            # for token in token_counter:
            #     if token not in inverted_index:
            #         inverted_index[token] = {}
            #     inverted_index[token][i] = token_counter[token]
            
            for bigram in bigram_counter_i:
                if bigram in self.bigram_set:
                    if bigram not in inverted_index:
                        inverted_index[bigram] = {}
                    inverted_index[bigram][i] = bigram_counter_i[bigram]
        
        #Pickle afterwards
        with open(f"{self.file_paths['inverted_index']}", 'wb') as f:  # open a text file
            pickle.dump(inverted_index, f) # serialize the list
            f.close()
        
        self.inverted_index = inverted_index
        self.tsed_DF = tsed_DF

    # Function to make embeddings. Right now only for 1 field.
    def make_embeddings(self, dataset):
        """
        make_embeddings

        Method to create a HuggingFace Dataset of embeddings from the "whole_func_string" column.
        CREDIT: Method written from HuggingFace tutorials.

        Parameters
        ----------
        dataset: HuggingFace dataset
            The dataset to train embeddings on
        """

        dataset_embeddings = dataset.map(
            lambda x: {"embeddings": get_embeddings(x["whole_func_string"]).detach().cpu().numpy()[0]}
        )

        dataset_embeddings.add_faiss_index(column="embeddings")

        with open(self.file_paths["embeddings_dataset"], 'wb') as f:  # open a text file
            pickle.dump(dataset_embeddings, f) # serialize the list
            f.close()

        self.embed_dataset = dataset_embeddings

    # FIXED
    def query_results_lc_naive_custom(self, query_string, k = 10, tf_alpha = 0.5, bigrams = True, kw_method = "TFIDF", bm_k = 1.2, bm_b = 0.75): 
        """
        query_results_lc_naive_custom

        A method of querying for results. It utilizes a linear combination of keyword matching and embedding cosine values.
        It is considered "naive" as it doesn't normalize and as so the keyword portion has more weight naturally.
        You can set the weights for the linear combination from 0-1.

        Parameters
        ----------
        query_string: str
            The string query
        
        k: int
            The k number of top results to return
        
        tf_alpha: float
            Float value from 0-1 which will set the linear combination ratio of the following equation
            (tf_alpha) * keyword_score + (1-tf_alpha) * cosine_score(query_embeddings, embeddings)

        bigrams: bool
            Boolean determining if bigrams want to be included in the search
        
        kw_method: str
            String either "TFIDF" or "BM25" that determines the keyword matching method.

        bm_k: float
            Float value that can be tuned for BM_25 calculation for saturation.

        bm_b: float
            Float value that can be tuned for BM_25 calculation for penalizing irrelevant tokens.
        """
        
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word.lower()) for word in tokenizer.tokenize(query_string) if word not in stop_words]
        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()

        # rel_indices = []
        answer_dict = {}

        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices = list(set(self.inverted_index[token].keys()))

                for rel_i in rel_indices:
                    if rel_i not in answer_dict:
                        answer_dict[rel_i] = [0,cosine_sim(query_embedding[0], self.tsed_DF["embeddings"][rel_i])]
                    tf = self.inverted_index[token][rel_i][0] / self.inverted_index[token][rel_i][1]
                    df = len(self.inverted_index[token])
                    idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                    if kw_method == "TFIDF":
                        answer_dict[rel_i][0] += tf * idf
                    else: #kw_method == "BM25"
                        bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[token][rel_i][1] / self.bm_avg_DL)))
                        answer_dict[rel_i][0] += bm_comp * idf

        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO", bigram, self.inverted_index[bigram])
                    rel_indices = list(set(self.inverted_index[bigram].keys()))
                    for rel_i in rel_indices:
                        if rel_i not in answer_dict:
                            answer_dict[rel_i] = [0,cosine_sim(query_embedding[0], self.tsed_DF["embeddings"][rel_i])]
                        tf = self.inverted_index[bigram][rel_i][0] / self.inverted_index[bigram][rel_i][1]
                        df = len(self.inverted_index[bigram])
                        idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                        
                        if kw_method == "TFIDF":
                            answer_dict[rel_i][0] += 2 * tf * idf
                        else:
                            bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[bigram][rel_i][1] / self.bm_avg_DL)))
                            answer_dict[rel_i][0] += 2 * bm_comp * idf

        result_lst = [[a,b] for (a,b) in answer_dict.items()]
        result_lst.sort(reverse=True, key = lambda x: x[1][0] * tf_alpha + x[1][1] * (1-tf_alpha))
        # print(result_lst[:k])
        return result_lst[:k]
        
    # FIXED
    def query_results_tfidf(self, query_string, k = 10, bigrams = False): #tf-idf JUST TF-IDF
        """
        query_results_lc_naive_custom

        A method of querying for results. It utilizes only TF-IDF

        Parameters
        ----------
        query_string: str
            The string query
        
        k: int
            The k number of top results to return

        bigrams: bool
            Boolean determining if bigrams want to be included in the search
        """
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word.lower()) for word in tokenizer.tokenize(query_string) if word not in stop_words]

        rel_indices = []
        answer_dict = {}

        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices = list(set(self.inverted_index[token].keys()))

                for rel_i in rel_indices:
                    if rel_i not in answer_dict:
                        answer_dict[rel_i] = 0
                    tf = self.inverted_index[token][rel_i][0] / self.inverted_index[token][rel_i][1]
                    df = len(self.inverted_index[token])
                    idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                    answer_dict[rel_i] += tf * idf

        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO", bigram, self.inverted_index[bigram])
                    rel_indices = list(set(self.inverted_index[bigram].keys()))
                    for rel_i in rel_indices:
                        if rel_i not in answer_dict:
                            answer_dict[rel_i] = 0
                        tf = self.inverted_index[bigram][rel_i][0] / self.inverted_index[bigram][rel_i][1]
                        df = len(self.inverted_index[bigram])
                        idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                        # print(2 * tf * idf)
                        answer_dict[rel_i] += 2 * tf * idf

        result_lst = [(a,b) for (a,b) in answer_dict.items()]
        result_lst.sort(reverse=True, key = lambda x: x[1])
        return result_lst[:k]
    
    # FIXED
    def query_results_BM25(self, query_string, k = 10, bm_k = 1.2, bm_b = 0.75, bigrams = False): 
        """
        query_results_lc_naive_custom

        A method of querying for results. It utilizes purely BM25.

        Parameters
        ----------
        query_string: str
            The string query
        
        k: int
            The k number of top results to return

        bigrams: bool
            Boolean determining if bigrams want to be included in the search
        
        bm_k: float
            Float value that can be tuned for BM_25 calculation for saturation.

        bm_b: float
            Float value that can be tuned for BM_25 calculation for penalizing irrelevant tokens.
        """
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word.lower()) for word in tokenizer.tokenize(query_string) if word not in stop_words]

        rel_indices = []
        answer_dict = {}

        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices = list(set(self.inverted_index[token].keys()))

                for rel_i in rel_indices:
                    if rel_i not in answer_dict:
                        answer_dict[rel_i] = 0
                    tf = self.inverted_index[token][rel_i][0] / self.inverted_index[token][rel_i][1]
                    df = len(self.inverted_index[token])
                    bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[token][rel_i][1] / self.bm_avg_DL)))

                    idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                    answer_dict[rel_i] += bm_comp * idf

        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO", bigram, self.inverted_index[bigram])
                    rel_indices = list(set(self.inverted_index[bigram].keys()))
                    for rel_i in rel_indices:
                        if rel_i not in answer_dict:
                            answer_dict[rel_i] = 0
                        tf = self.inverted_index[bigram][rel_i][0] / self.inverted_index[bigram][rel_i][1]
                        df = len(self.inverted_index[bigram])
                        idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                        bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[bigram][rel_i][1] / self.bm_avg_DL)))
                        answer_dict[rel_i] += 2 * bm_comp * idf

        result_lst = [(a,b) for (a,b) in answer_dict.items()]
        result_lst.sort(reverse=True, key = lambda x: x[1])
        return result_lst[:k]

    # FIXED
    def query_results_embed(self, query_string, k = 10): #just embed faiss
        """
        query_results_lc_naive_custom

        A method of querying for results. It FAISS nearest matches and takes the top
        scores from that result.

        Parameters
        ----------
        query_string: str
            The string query
        
        k: int
            The k number of top results to return
        """
        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        desc_scores, desc_results = self.embed_dataset.search("embeddings", query_embedding, k)
        
        
        result_lst = [[a,b] for (a,b) in zip(desc_results, desc_scores)]
        result_lst.sort(reverse=True, key = lambda x: x[1])
        return result_lst[:k]

    # FIXED
    def query_results_lc_norm(self, query_string, k = 10, bigrams = False, tf_alpha = 0.5,kw_method = "TFIDF", bm_k = 1.2, bm_b = 0.75): 
        """
        query_results_lc_norm

        A method of querying for results. It utilizes a linear combination of keyword matching and embedding cosine values.
        It normalizes and as so the keyword portion is constrained to 0-1.
        You can set the weights for the linear combination from 0-1.

        Parameters
        ----------
        query_string: str
            The string query
        
        k: int
            The k number of top results to return
        
        tf_alpha: float
            Float value from 0-1 which will set the linear combination ratio of the following equation
            (tf_alpha) * keyword_score + (1-tf_alpha) * cosine_score(query_embeddings, embeddings)

        bigrams: bool
            Boolean determining if bigrams want to be included in the search
        
        kw_method: str
            String either "TFIDF" or "BM25" that determines the keyword matching method.

        bm_k: float
            Float value that can be tuned for BM_25 calculation for saturation.

        bm_b: float
            Float value that can be tuned for BM_25 calculation for penalizing irrelevant tokens.
        """
        
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word.lower()) for word in tokenizer.tokenize(query_string) if word not in stop_words]
        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()

        # rel_indices = []
        answer_dict = {}

        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices = list(set(self.inverted_index[token].keys()))

                for rel_i in rel_indices:
                    if rel_i not in answer_dict:
                        answer_dict[rel_i] = [0,cosine_sim(query_embedding[0], self.tsed_DF["embeddings"][rel_i])]
                    tf = self.inverted_index[token][rel_i][0] / self.inverted_index[token][rel_i][1]
                    df = len(self.inverted_index[token])
                    idf = np.log((len(self.tsed_DF) + 1) / (df + 1))

                    if kw_method == "TFIDF":
                        answer_dict[rel_i][0] += tf * idf
                    else:
                        bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[token][rel_i][1] / self.bm_avg_DL)))
                        answer_dict[rel_i][0] += bm_comp * idf

        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO", bigram, self.inverted_index[bigram])
                    rel_indices = list(set(self.inverted_index[bigram].keys()))
                    for rel_i in rel_indices:
                        if rel_i not in answer_dict:
                            answer_dict[rel_i] = [0,cosine_sim(query_embedding[0], self.tsed_DF["embeddings"][rel_i])]
                        tf = self.inverted_index[bigram][rel_i][0] / self.inverted_index[bigram][rel_i][1]
                        df = len(self.inverted_index[bigram])
                        idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                        
                        if kw_method == "TFIDF":
                            answer_dict[rel_i][0] += 2 * tf * idf
                        else:
                            bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[bigram][rel_i][1] / self.bm_avg_DL)))
                            answer_dict[rel_i][0] += 2 * bm_comp * idf

        result_lst = [[a,b] for (a,b) in answer_dict.items()]

        
        if len(result_lst) > 0:
            max_tf_idf_score = max([x[1][0] for x in result_lst])
        else: 
            max_tf_idf_score = 1

        result_lst.sort(reverse=True, key = lambda x: x[1][0] * (tf_alpha) / max_tf_idf_score + x[1][1] * (1 - tf_alpha))
        # print(result_lst[:k])
        return result_lst[:k]
        
    # FIXED
    def query_results_odds_evens(self, query_string, k = 10, bigrams = False, kw_method = "TFIDF", bm_k = 1.2, bm_b = 0.75): 
        """
        query_results_odds_evens

        A method of querying for the results using both keyword matching and semantic embeddings FAISS. The
        top results are then put in alternating order (semantic results are evens, keyword results are odds)

        Parameters
        ----------
        query_string: str
            The string query
        
        k: int
            The k number of top results to return

        bigrams: bool
            Boolean determining if bigrams want to be included in the search
        
        kw_method: str
            String either being "TFIDF" or "BM25" that sets the keyword matching algorithm
        
        bm_k: float
            Float value that can be tuned for BM_25 calculation for saturation.

        bm_b: float
            Float value that can be tuned for BM_25 calculation for penalizing irrelevant tokens.
        """

        #Embedding portion
        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        desc_scores, desc_results = self.embed_dataset.search("embeddings", query_embedding, k)
        
        result_lst_embed = [[a,b] for (a,b) in zip(desc_results, desc_scores)]
        result_lst_embed.sort(reverse=True, key = lambda x: x[1])

        embed_top_results = result_lst_embed

        #TF-IDF Portion

        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word.lower()) for word in tokenizer.tokenize(query_string) if word not in stop_words]

        rel_indices = []
        answer_dict = {}

        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices = list(set(self.inverted_index[token].keys()))

                for rel_i in rel_indices:
                    if rel_i not in answer_dict:
                        answer_dict[rel_i] = 0
                    tf = self.inverted_index[token][rel_i][0] / self.inverted_index[token][rel_i][1]
                    df = len(self.inverted_index[token])
                    idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                    if kw_method == "TFIDF":
                        answer_dict[rel_i] += tf * idf
                    else: 
                        bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[token][rel_i][1] / self.bm_avg_DL)))
                        answer_dict[rel_i] += bm_comp * idf

        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                   
                    rel_indices = list(set(self.inverted_index[bigram].keys()))
                    for rel_i in rel_indices:
                        if rel_i not in answer_dict:
                            answer_dict[rel_i] = 0
                        tf = self.inverted_index[bigram][rel_i][0] / self.inverted_index[bigram][rel_i][1]
                        df = len(self.inverted_index[bigram])
                        idf = np.log((len(self.tsed_DF) + 1) / (df + 1))
                        if kw_method == "TFIDF":
                            answer_dict[rel_i] += 2 * tf * idf
                        else:
                            bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[bigram][rel_i][1] / self.bm_avg_DL)))
                            answer_dict[rel_i] += 2 * bm_comp * idf

        result_lst_tfidf = [(a,b) for (a,b) in answer_dict.items()]
        result_lst_tfidf.sort(reverse=True, key = lambda x: x[1])
        tfidf_top_results = result_lst_tfidf

        #Make sure no duplicate indices
        result_lst = []
        set_indices = set()
        i_embed = 0
        i_tfidf = 0
        # print(embed_top_results, tfidf_top_results)
        for i in range(k):
            if i % 2 == 0: #0,2,4 are embed
                while i_embed < len(embed_top_results) and embed_top_results[i_embed][0] in set_indices:
                    i_embed += 1
                if i_embed < len(embed_top_results):
                    result_lst.append(embed_top_results[i_embed])
                    set_indices.add(embed_top_results[i_embed][0])
                    i_embed += 1
            else:
                while i_tfidf < len(tfidf_top_results) and tfidf_top_results[i_tfidf][0] in set_indices:
                    i_tfidf += 1
                if i_tfidf < len(tfidf_top_results):
                    result_lst.append(tfidf_top_results[i_tfidf])
                    set_indices.add(tfidf_top_results[i_tfidf][0])
                    i_tfidf += 1
        
        # print(result_lst)
        return result_lst

    # FIXED
    def query_results_faiss_kw(self, query_string, k = 10, bigrams = False, kw_method = "TFIDF", bm_k = 1.2, bm_b = 0.75):
        """
        query_results_faiss_kw

        A method of querying for the results using both keyword matching and semantic embeddings FAISS. The results
        are fist obtained with FAISS, then reorded with keyword ranking.

        Parameters
        ----------
        query_string: str
            The string query
        
        k: int
            The k number of top results to return

        bigrams: bool
            Boolean determining if bigrams want to be included in the search
        
        kw_method: str
            String either being "TFIDF" or "BM25" that sets the keyword matching algorithm
        
        bm_k: float
            Float value that can be tuned for BM_25 calculation for saturation.

        bm_b: float
            Float value that can be tuned for BM_25 calculation for penalizing irrelevant tokens.
        """
         
        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        _, desc_results = self.embed_dataset.search("embeddings", query_embedding, k)

        # desc_scores are the indices for the "closest neighbors to the query"
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word) for word in tokenizer.tokenize(query_string) if word not in stop_words]

        
        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO")
                    query_tokens.append(bigram)
        
        result_lst = []
        for i in desc_results:
            kw_score = 0
            for token in query_tokens:
                
                if token in self.inverted_index and i in self.inverted_index[token]:
                    tf = self.inverted_index[token][i][0] / self.inverted_index[token][i][0]
                    df = len(self.inverted_index[token])
                    idf = np.log((len(self.tsed_DF) + 1) / (df + 1))

                    if kw_method == "TFIDF":
                        kw_score += tf * idf
                    else:
                        bm_comp = (tf * (bm_k + 1)) / (tf + bm_k * (1 - bm_b + bm_b * (self.inverted_index[token][i][1] / self.bm_avg_DL)))
                        kw_score += bm_comp * idf
            
            result_lst.append([i, kw_score])
        
        result_lst.sort(reverse=True, key = lambda x: x[1])
        # print(result_lst)
        return result_lst[:k]

    # FIXED
    def query_results_faiss_cos(self, query_string, k = 10, bigrams = False):
        """
        query_results_faiss_cos

        A method of querying for the results using only semantic embeddings. The results
        are fist obtained with FAISS, then reorded with cosine similarities of the query embeddings and the result
        embeddings.

        Parameters
        ----------
        query_string: str
            The string query
        
        k: int
            The k number of top results to return

        bigrams: bool
            Boolean determining if bigrams want to be included in the search
        """

        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        _, desc_results = self.embed_dataset.search("embeddings", query_embedding, 2 * k)

        # desc_scores are the indices for the "closest neighbors to the query"
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word) for word in tokenizer.tokenize(query_string) if word not in stop_words]

        
        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO", bigram)
                    query_tokens.append(bigram)
                    
        
        result_lst = []
        for i in desc_results:
            result_lst.append([i, cosine_sim(self.tsed_DF["embeddings"][i], query_embedding[0])])

        
        result_lst.sort(reverse=True, key = lambda x: x[1])
        return result_lst[:k]


    # Function which runs all 99 queries, and returns a pd df of the results
    def create_results(self, query_filepath, results_per_query = 100):
        """
        create_results:
        Function that takes in the 99 test queries and returns it as a Pandas DataFrame
        
        Parameters
        ----------
        query_filepath: str
            The string which locates where the 99 preset queries are
        
        results_per_query: int
            The integer amount of top results to be returned per the query.
        """

        queries = pd.read_csv(query_filepath)
        q_lst = queries["query"].to_list()

        lang_lst = []
        func_code_url_lst = []
        query_lst = []
        func_docs_lst = []

        for i, query in enumerate(q_lst):
            # print(i)
            fbm_lst = self.query_results_lc_naive_custom(query, results_per_query, kw_method="BM25", tf_alpha=0.75,bigrams=True) #CHANGE THIS LINE TO CHECK DIFFERENT METHODS. FALSE = NO BIGRAMS
            query_lst += [query for j in range(len(fbm_lst))]
            
            for lst in fbm_lst:
                # print(lst)
                lang_lst.append(self.tsed_DF.iloc[lst[0]]["language"])
                func_code_url_lst.append(self.tsed_DF.iloc[lst[0]]["func_code_url"])
                func_docs_lst.append(self.tsed_DF.iloc[lst[0]]['func_documentation_string'])
                # func_names_lst.append(self.tsed_DF.iloc[lst[0]]['func_name'])


        prediction_df = pd.DataFrame({'language' : lang_lst, 'url': func_code_url_lst, "query" : query_lst, "documentation" : func_docs_lst})
        return prediction_df