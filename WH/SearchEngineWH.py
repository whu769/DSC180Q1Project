#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
# from num2words import num2words


# In[ ]:





# In[2]:


#CREDIT
# GOT AIRBNB DATA FROM: https://www.kaggle.com/datasets/alexanderfreberg/airbnb-listings-2016-dataset/data

# Followed guide on TF-IDF search. On this
# https://github.com/williamscott701/Information-Retrieval/blob/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score/TF-IDF.ipynb
# https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089


# In[3]:


data_listing = pd.read_csv("Data/TFP_Listing.csv")
data_review = pd.read_csv("Data/TFP_Reviews.csv")


# In[4]:


data_listing.head()


# In[5]:


data_review.head()


# In[6]:


# data_listing["description"][1]
# data_listing["summary"][1]
# data_listing["space"][1]


# In[7]:


data_listing["name"].isnull().sum()


# In[8]:


data_listing["space"].isnull().sum()


# In[9]:


### Process the text. Process the space text as it seems the most text heavy?
def process_text(s):
    # lower case, keep only letters numbers and spaces
    # eliminate double spaces
   
    s = s.lower()
    s = re.sub(r'[^0-9a-zA-Z\s-]+', ' ', s)
    s = re.sub(r'[\s]{2,}', " ", s)
    
    words_lst = s.split()
    
    #stem
    stemmer = PorterStemmer()
    
    new_s = ""
    for word in words_lst:
        if len(word) > 1:
            new_s += " " + stemmer.stem(word)
    
    return new_s
            


# In[10]:


# Creating inverted index based off this article: https://www.geeksforgeeks.org/inverted-index/
def make_documents(data, col_name):
    documents = data[col_name].dropna().apply(process_text).to_dict()
    return documents

def make_inverted_index(documents):
    word_array = np.array(list(documents.values()))
    all_words = []
    for words in word_array:
        all_words +=  words.split(" ")
#     terms = dict(zip( range(len(set(all_words))),set(all_words)))
#     return terms
    all_words = set(all_words)
    inverted_index = {}
    
    for word in all_words:
        if word != "":
            lst_docs = []
            for i, doc in documents.items():
                if word in doc.split():
                    lst_docs.append(i)
        
            inverted_index[word] = lst_docs
    return inverted_index
    


# In[11]:


# documents = make_documents(data_listing, "space")
# documents
# data_listing["processed_space"] = data_listing["space"].apply(process_text)


# In[12]:


# documents
# inverted_index = make_inverted_index(documents)


# In[13]:


# inverted_index


# In[14]:


# total_vocab = [x for x in inverted_index]
# total_vocab


# In[15]:


def make_tfidf_DF(documents, inverted_index, total_vocab):
    tf_idf = {}
    df = pd.DataFrame()
    for i, doc in documents.items():
        term_lst = []
        for term in total_vocab:
            doc_lst = doc.split()
            tf = doc_lst.count(term) / len(doc_lst)

            idf = np.log(len(documents) / len(inverted_index[term]))
    #         if tf*idf > 0:
    #             print(tf*idf)
    #             print(term)
            term_lst.append(tf*idf)
            tf_idf[i, term] = tf*idf
        df[i] = term_lst
    return df
    
    
    


# In[16]:


# documents[0]


# In[17]:


# tf_idf[0, "annex"]


# In[18]:


# tf_idf
# len(tf_idf)


# In[19]:


# df_tfidf = df.copy()
# df_index = pd.Series(total_vocab)
# df_tfidf = df_tfidf.set_index(df_index)
# df_tfidf


# In[20]:


# 8343 * 3249


# In[21]:


# total_vocab


# In[ ]:





# In[22]:


#Got from William Scott https://github.com/williamscott701/Information-Retrieval/blob/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score/TF-IDF.ipynb
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


# In[23]:


# def find_best_matches(query, k):
#     q_vector = process_query(query)
    
#     cosine_lst = []
#     for x in df_tfidf.columns.to_list():
#         col = df_tfidf[x].to_numpy()
        
#         cosine_lst.append((cosine_sim(q_vector, col), x))
    
#     cosine_lst.sort(reverse = True, key = lambda x: x[0])
#     return cosine_lst[:k]


# In[24]:


# cosine_sim(q_vector, df_tfidf[3814].to_numpy())
# cos_lst = find_best_matches("Queen anne hill. Single. Free wifi and laundry", 5)


# In[25]:


# df_tfidf.columns.to_list()
# cos_lst


# In[26]:


# for tup in cos_lst:
#     print(data_listing["space"].iloc[tup[1]])
#     print()


# In[ ]:





# In[ ]:





# In[27]:


#Attempt at including both documents and title
#Make the documents
documents_space = make_documents(data_listing, "space")
documents_title = make_documents(data_listing, "name")


# In[28]:


#Make the inverted indices
inverted_index_space = make_inverted_index(documents_space)
inverted_index_title = make_inverted_index(documents_title)


# In[29]:


#Total vocab
total_vocab_space = [x for x in inverted_index_space]
total_vocab_title = [x for x in inverted_index_title]


# In[30]:


space_tfidf_DF = make_tfidf_DF(documents_space, inverted_index_space, total_vocab_space)


# In[31]:


space_tfidf_DF.shape


# In[32]:


title_tfidf_DF = make_tfidf_DF(documents_title, inverted_index_title, total_vocab_title)


# In[33]:


title_tfidf_DF.shape


# In[34]:


#Process query. Make it into a vector of tf-idfs
def process_query(s, inverted_index, total_vocab, documents):
    processed_s = process_text(s)
#     print(processed_s)
    lst_words = processed_s.split()
#     print(lst_words)
    q = np.zeros(len(total_vocab))
#     print(len(q))
    counter = Counter(lst_words)
    for word in np.unique(lst_words):
        if word in inverted_index:
            tf = counter[word] / len(lst_words)
            df = len(inverted_index[word])
            idf = np.log(len(documents) / df)
            q[total_vocab.index(word)] = tf*idf
    
    return q


# In[35]:


def combine_tfidf_grading(cosine_lst_space, cosine_lst_title, alpha):
    combined_score = []
    for i in cosine_lst_title:
        cls_item = cosine_lst_space[i[1]]
#         print(cls_item)
        if isinstance(cls_item, list):
            combined_score.append([i[0] * alpha + cls_item[0] * (1-alpha), i[1]])
        else:
            combined_score.append([i[0] * alpha, i[1]])
    return combined_score


# In[36]:


def find_best_matches(query, k, alpha = 0.5):
    q_vector_title = process_query(query, inverted_index_title, total_vocab_title, documents_title)
    q_vector_space = process_query(query, inverted_index_space, total_vocab_space, documents_space)
#     print(len(q_vector_space))
#     print(len(q_vector_title))
    
    cosine_lst_title = [0 for x in range(len(title_tfidf_DF.columns.to_list()))]
    cosine_lst_space = [0 for x in range(len(title_tfidf_DF.columns.to_list()))]
#     print(len(cosine_lst_title))
#     print(len(cosine_lst_space))
    
    for x in space_tfidf_DF.columns.to_list():
        col = space_tfidf_DF[x].to_numpy()

        cosine_lst_space[x] = [cosine_sim(q_vector_space, col), x]
    
    for x in title_tfidf_DF.columns.to_list():
        col = title_tfidf_DF[x].to_numpy()

        cosine_lst_title[x] = [cosine_sim(q_vector_title, col),x]
    
#     cosine_lst_space, cosine_lst_title
    comb_score = combine_tfidf_grading(cosine_lst_space, cosine_lst_title, alpha) 
    comb_score.sort(reverse = True, key = lambda x: x[0])
    return comb_score[:k]


# In[37]:


results = find_best_matches("Queen anne hill. Single. Free wifi Privacy", 5, 0.8)


# In[38]:


results


# In[39]:


for x in results:
    index = x[1]
    val = data_listing.iloc[index]
    print(val["name"])
    print(val["space"])
    print("-----------------------------------------------------------------")


# In[40]:


# cos_lst_title[:10]


# In[ ]:





# In[41]:


# combined_score = combine_tfidf_grading()


# In[ ]:


# len(combined_score)


# In[ ]:


# combined_score.sort(reverse = True, key = lambda x: x[0])
# combined_score[:10]


# In[42]:


data_listing[["name", "space"]].iloc[86]["name"]


# In[43]:


test_query1 = "Beautiful Ballard condo with view! Beautiful condo facing south and west with an open living room/kitchen area and 2 spacious bedrooms \
(ceiling fans included!).  Main living room has a view of Mt. Rainier and the Olympics and offers front-row views to \
amazing sunsets. Comfortable sleeping for four people with an option for an air mattress to add 1 or 2 more people.\
I live here most of the time and so the space comes with a fully equipped kitchen for all your cooking needs, \
including a coffee-maker and grinder, toaster, stove/oven, dishwasher, and all other basic kitchen amenities \
(minus a microwave). There is a deck as well with a grill for you to use and 2 or 3 chairs to relax in."


# In[44]:


results1 = find_best_matches(test_query1, 5, 0.5)


# In[45]:


for x in results1:
    index = x[1]
    val = data_listing.iloc[index]
    print(val["name"])
    print(val["space"])
    print("-----------------------------------------------------------------")


# In[ ]:


test_query2 = "Private single Bothell"
results2 = find_best_matches(test_query2, 5, 0.5)


# In[ ]:


for x in results2:
    index = x[1]
    val = data_listing.iloc[index]
    print(val["name"])
    print(val["space"])
    print("-----------------------------------------------------------------")


# In[ ]:


test_query3 = "Shared house in Capitol Hill, easy transportation!"
results3 = find_best_matches(test_query3, 5, 0.5)


# In[ ]:


for x in results3:
    index = x[1]
    val = data_listing.iloc[index]
    print(val["name"])
    print(val["space"])
    print("-----------------------------------------------------------------")


# In[ ]:


# Evaluation: Works with basically the same exact description
# Test_query 2: Not the best option? In terms of location. Not sure if Bothell exists in the document.
# Test_query 3: Defo the ideal option was probably option 3. Not the best results.


# In[ ]:




