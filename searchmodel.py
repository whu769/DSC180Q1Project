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
hg_model = "huggingface/CodeBERTa-small-v1" #"sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_ckpt = hg_model #Can/Should test different models
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
trained_model = AutoModel.from_pretrained(model_ckpt)
trained_model.to(device)

#Tokenizer initialization
st = RegexpStemmer('ing$|s$|e$|able$', min=4)
rgx_tokenizer = RegexpTokenizer(r'\w+')

class searchmodel:

    def __init__(self, data, file_paths, loadEmbed = False, loadTF = False):
        self.data = data
        self.loadEmbed = loadEmbed
        self.loadTF = loadTF
        self.file_paths = file_paths #{"inverted_index" : "path_to_inverted_index", "embeddings_dataset" : "path_to_embedded_dataset"}

        self.inverted_index = None
        self.embed_dataset = None

        if loadEmbed:
            with open(file_paths["embeddings_dataset"], 'rb') as f:  # open a text file
                self.embed_dataset = pickle.load(f) # serialize the list
                f.close()
            
            self.tsed_DF = self.embed_dataset.to_pandas()
        else:
            self.make_embeddings(self.data)
            # self.make_inverted_index_docs()

        if loadTF: #if the load boolean is true, file_paths is a dictionary to load info
            with open(file_paths["inverted_index"], 'rb') as f:  # open a text file
                self.inverted_index = pickle.load(f) # serialize the list
                f.close()
        else:
            # self.make_inverted_index_docs()
            self.make_inverted_index_docs_bigrams()
            # self.make_inverted_index_docs_bigrams_only()
                
       
        stop_words = set(stopwords.words('english'))
        #self.tsed_DF["clean_code_tokens"] =  self.tsed_DF["func_code_tokens"].apply(clean_code_tokens)
        self.tsed_DF["func_doc_tokens"] = self.tsed_DF["func_documentation_string"].apply(lambda x: rgx_tokenizer.tokenize(x))
        self.tsed_DF["func_doc_stem_tokens"] = self.tsed_DF["func_doc_tokens"].apply(lambda x: [st.stem(word) for word in x if word not in stop_words])
        self.create_tfidf(bigrams=True) # SWITCH THIS BACK
        # self.create_tfidf_bigrams_only(bigrams=True) #SWITCH TO TRUE FOR BIGRAMS

    #Legacy 
    def make_inverted_index_code(self):
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

    # No bigrams
    def make_inverted_index_docs(self):
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
            token_counter = Counter(tsed_DF.iloc[i]["func_doc_stem_tokens"])
            bigram_counter_i = Counter(list(ngrams(tsed_DF.iloc[i]["func_doc_stem_tokens"], 2)))

            for token in token_counter:
                if token not in inverted_index:
                    inverted_index[token] = {}
                inverted_index[token][i] = token_counter[token]
            
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
    
    def make_inverted_index_docs_bigrams_only(self):
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

    def create_tfidf(self, column = "func_doc_stem_tokens", bigrams = False):
        num_rows = len(self.tsed_DF)

        tf_idf = {}
        for i in range(num_rows):
        # print(i)
            tokens = self.tsed_DF[column].iloc[i]
            if bigrams:
                bigram_lst_i = []
                for bigram in list(ngrams(self.tsed_DF.iloc[i]["func_doc_stem_tokens"], 2)):
                    if bigram in self.bigram_set:
                        bigram_lst_i.append(bigram)
                tokens += bigram_lst_i

            counter = Counter(tokens)
            words_count = len(tokens)

            for token in set(tokens):
                tf = counter[token] / words_count
                df = len(self.inverted_index[token])
                idf = np.log((num_rows + 1) / (df + 1))

                tf_idf[i, token] = tf * idf
        self.tf_idf = tf_idf
    
    def create_tfidf_bigrams_only(self, column = "func_doc_stem_tokens", bigrams = False):
        num_rows = len(self.tsed_DF)

        tf_idf = {}
        for i in range(num_rows):
        # print(i)
            tokens = []#self.tsed_DF[column].iloc[i] Only bigrams lmao
            if bigrams:
                bigram_lst_i = []
                for bigram in list(ngrams(self.tsed_DF.iloc[i]["func_doc_stem_tokens"], 2)):
                    if bigram in self.bigram_set:
                        bigram_lst_i.append(bigram)
                tokens += bigram_lst_i

            counter = Counter(tokens)
            words_count = len(tokens)

            for token in set(tokens):
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
            lambda x: {"embeddings": get_embeddings(x["whole_func_string"]).detach().cpu().numpy()[0]}
        )

        dataset_embeddings.add_faiss_index(column="embeddings")

        with open(self.file_paths["embeddings_dataset"], 'wb') as f:  # open a text file
            pickle.dump(dataset_embeddings, f) # serialize the list
            f.close()

        self.embed_dataset = dataset_embeddings

    def query_results_lc_naive_custom(self, query_string, k = 10, tf_alpha = 0.75, bigrams = True): #lc = Linear-combination, Implicitly weighted towards TF-IDF
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word) for word in tokenizer.tokenize(query_string) if word not in stop_words]


        rel_indices = []
        
        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices += list(self.inverted_index[token].keys())
        
        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO")
                    query_tokens.append(bigram)
                    rel_indices += list(self.inverted_index[bigram].keys())
        
        rel_indices = set(rel_indices)

        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        # len(query_embedding[0])
        # len(tsed_DF["embeddings"][0])
        
        result_lst = []
        for i in rel_indices:
            tf_score = 0
            for token in query_tokens:
                if (i, token) in self.tf_idf:
                    if type(token) != str:
                        # print("HELLO", token)
                        tf_score += 3 * (self.tf_idf[(i, token)])
                    else:
                        tf_score += (self.tf_idf[(i, token)])

            result_lst.append([i, tf_score, cosine_sim(self.tsed_DF["embeddings"][i], query_embedding[0])])
        
        result_lst.sort(reverse=True, key = lambda x: tf_alpha * x[1] + (1 - tf_alpha)*x[2])
        return result_lst[:k]

    
    # FIXED
    def query_results_lc_naive(self, query_string, k = 10, bigrams = True): #lc = Linear-combination, Implicitly weighted towards TF-IDF
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word) for word in tokenizer.tokenize(query_string) if word not in stop_words]


        rel_indices = []
        
        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices += list(self.inverted_index[token].keys())
        
        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO")
                    query_tokens.append(bigram)
                    rel_indices += list(self.inverted_index[bigram].keys())
        
        rel_indices = set(rel_indices)

        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        # len(query_embedding[0])
        # len(tsed_DF["embeddings"][0])
        
        result_lst = []
        for i in rel_indices:
            tf_score = 0
            for token in query_tokens:
                if (i, token) in self.tf_idf:
                    if type(token) != str:
                        tf_score += 2 * (self.tf_idf[(i, token)])
                    else:
                        tf_score += (self.tf_idf[(i, token)])

            result_lst.append([i, tf_score, cosine_sim(self.tsed_DF["embeddings"][i], query_embedding[0])])
        
        result_lst.sort(reverse=True, key = lambda x: 0.5 * x[1] + 0.5*x[2])
        return result_lst[:k]

    # FIXED
    def query_results_tfidf(self, query_string, k = 10, bigrams = False): #tf-idf JUST TF-IDF

        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word) for word in tokenizer.tokenize(query_string) if word not in stop_words]

        rel_indices = []
        
        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices += list(self.inverted_index[token].keys())

        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO")
                    query_tokens.append(bigram)
                    rel_indices += list(self.inverted_index[bigram].keys())
        
        rel_indices = set(rel_indices)

        result_lst = []
        for i in rel_indices:
            tf_score = 0
            for token in query_tokens:
                
                if (i, token) in self.tf_idf:
                    if type(token) != str:
                        tf_score += 2 * (self.tf_idf[(i, token)])
                    
                    #if you comment out the else portion, it only checks bigrams lol
                    # else:
                    #     tf_score += (self.tf_idf[(i, token)])
            
            result_lst.append([i, tf_score])
        
        result_lst.sort(reverse=True, key = lambda x: x[1])
        return result_lst[:k]
    
    # Don't need to fix
    def query_results_embed(self, query_string, k = 10): #tf-idf JUST TF-IDF

        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        desc_scores, desc_results = self.embed_dataset.search("embeddings", query_embedding, k)
        
        
        result_lst = [[a,b] for (a,b) in zip(desc_results, desc_scores)]
        result_lst.sort(reverse=True, key = lambda x: x[1])
        return result_lst[:k]

    # FIXED
    def query_results_lc_norm(self, query_string, k = 10, bigrams = False): #lc = normalize tf-idf score so that and cosine are all between 0-1 and so will the final result
        # query_tokens = query_string.split()
        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word) for word in tokenizer.tokenize(query_string) if word not in stop_words]

        rel_indices = []
        
        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices += list(self.inverted_index[token].keys())
        
        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO")
                    query_tokens.append(bigram)
                    rel_indices += list(self.inverted_index[bigram].keys())
        
        rel_indices = set(rel_indices)

        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()

        result_lst = []
        for i in rel_indices:
            tf_score = 0
            for token in query_tokens:
                
                if (i, token) in self.tf_idf:
                    if type(token) != str:
                        tf_score += 2 * (self.tf_idf[(i, token)])
                    else:
                        tf_score += (self.tf_idf[(i, token)])
                # try:
                #     tf_score += (self.tf_idf[(i, token)])
                # except: continue #this is bad, make sure this isn't the play
            # print(i)

            result_lst.append([i, tf_score, cosine_sim(self.tsed_DF["embeddings"][i], query_embedding[0])])
        
        #normalize tf_score
        if len(result_lst) > 0:
            tf_scores = [x[1] for x in result_lst]
            # print(tf_scores)
            max_tf = max(tf_scores)
            if max_tf > 0:
                tf_scores = [x / max_tf for x in tf_scores]
                for i, tf_s in enumerate(tf_scores):
                    result_lst[i][1] = tf_s
        
        result_lst.sort(reverse=True, key = lambda x: 0.5 * x[1] + 0.5*x[2])
        return result_lst[:k]

    # FIXED
    def query_results_odds_evens(self, query_string, k = 10, bigrams = False): #do TF-IDF, do FAISS. Take the top k/2 for both. Make sure no overlap
        #So long as return result_lst[0] is the index. It'll be chill

        #Embedding portion
        query_embedding = get_embeddings([query_string]).cpu().detach().numpy()
        desc_scores, desc_results = self.embed_dataset.search("embeddings", query_embedding, k)
        
        result_lst_embed = [[a,b] for (a,b) in zip(desc_results, desc_scores)]
        result_lst_embed.sort(reverse=True, key = lambda x: x[1])

        embed_top_results = result_lst_embed

        #TF-IDF Portion

        stop_words = set(stopwords.words('english'))
        query_tokens = [st.stem(word) for word in tokenizer.tokenize(query_string) if word not in stop_words]

        rel_indices = []
        
        for token in query_tokens:
            if token in self.inverted_index:
                rel_indices += list(self.inverted_index[token].keys())
        
        if bigrams:
            bigram_lst = list(ngrams(query_tokens, 2))
            for bigram in bigram_lst:
                if bigram in self.inverted_index:
                    # print("HELLOOOOOOOOOOO")
                    query_tokens.append(bigram)
                    rel_indices += list(self.inverted_index[bigram].keys())

        rel_indices = set(rel_indices)

        result_lst_tfidf = []
        for i in rel_indices:
            tf_score = 0
            for token in query_tokens:
                if (i, token) in self.tf_idf:
                    if type(token) != str:
                        tf_score += 2 * (self.tf_idf[(i, token)])
                    else:
                        tf_score += (self.tf_idf[(i, token)])

            result_lst_tfidf.append([i, tf_score])
        
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

    def query_results_faiss_tf(self, query_string, k = 10, bigrams = False):
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
            tf_score = 0
            for token in query_tokens:
                
                if (i, token) in self.tf_idf:
                    if type(token) != str:
                        tf_score += 2 * (self.tf_idf[(i, token)])
                    else:
                        tf_score += (self.tf_idf[(i, token)])
            
            result_lst.append([i, tf_score])
        
        result_lst.sort(reverse=True, key = lambda x: x[1])
        # print(result_lst)
        return result_lst[:k]


    def query_results_faiss_cos(self, query_string, k = 10, bigrams = False):
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
            # tf_score = 0
            # for token in query_tokens:
            #     if (i, bigram) in self.tf_idf:
            #         print(self.tf_idf[(i, token)])
            #         tf_score += (self.tf_idf[(i, token)])
            
            # if tf_score != 0:
            #     print(tf_score)

            result_lst.append([i, cosine_sim(self.tsed_DF["embeddings"][i], query_embedding[0])])

        
        result_lst.sort(reverse=True, key = lambda x: x[1])
        # print(result_lst)
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
            fbm_lst = self.query_results_tfidf(query, results_per_query, bigrams=True) #CHANGE THIS LINE TO CHECK DIFFERENT METHODS. FALSE = NO BIGRAMS
            query_lst += [query for j in range(len(fbm_lst))]
            
            for lst in fbm_lst:
                # print(lst)
                lang_lst.append(self.tsed_DF.iloc[lst[0]]["language"])
                func_code_url_lst.append(self.tsed_DF.iloc[lst[0]]["func_code_url"])

        prediction_df = pd.DataFrame({'language' : lang_lst, 'url': func_code_url_lst, "query" : query_lst})
        return prediction_df