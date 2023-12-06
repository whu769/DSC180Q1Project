#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, TFAutoModel
from datasets import Dataset


# In[ ]:


# !pip install datasets evaluate transformers[sentencepiece]
# !pip install faiss-gpu


# In[ ]:


#Credit: https://huggingface.co/learn/nlp-course/chapter5/6?fw=tf
# Colab Code: https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_tf.ipynb#scrollTo=7TMHFbt9maFp


# In[2]:


data_listing = pd.read_csv("Data/TFP_Listing.csv")
data_review = pd.read_csv("Data/TFP_Reviews.csv")


# In[ ]:


# data_listing["description"].isnull().sum() == data_listing["name"].isnull().sum()
# All are 0 


# In[3]:


data_listing.head()


# In[ ]:





# In[4]:


model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)


# In[5]:


#From Hugging Face Tutorials
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


# In[6]:


embedding = get_embeddings([data_listing["space"][0]])
embedding.shape


# In[ ]:


# embedding


# In[ ]:


# data_listing["desc_embeddings"] = data_listing["description"].apply(lambda x: get_embeddings([x]))


# In[ ]:


# data_listing["desc_embeddings"][0]


# In[ ]:


# data_listing.to_pickle("./Data/data_listing.pkl")


# In[ ]:


# data_listing["desc_embeddings"][1]


# In[7]:


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


# In[ ]:





# In[ ]:





# In[ ]:


airbnb_dataset = Dataset.from_pandas(data_listing)


# In[ ]:


airbnb_dataset


# In[ ]:


#Code from the colab
embeddings_dataset = airbnb_dataset.map(
    lambda x: {"desc_embeddings": get_embeddings(x["description"]).numpy()[0],
              "name_embeddings": get_embeddings(x["name"]).numpy()[0]}
)


# In[ ]:


#Code from the colab
embeddings_dataset.add_faiss_index(column="desc_embeddings")
embeddings_dataset.add_faiss_index(column="name_embeddings")


# In[ ]:





# In[8]:


# Code from: https://www.datacamp.com/tutorial/pickle-python-tutorial

# with open('./Data/embeddings_dataset.pkl', 'wb') as f:  # open a text file
#     pickle.dump(embeddings_dataset, f) # serialize the list
#     f.close()


with open('./Data/embeddings_dataset.pkl', 'rb') as f:

    embeddings_dataset = pickle.load(f) # deserialize using load()
    print(embeddings_dataset) # print student names
    f.close()


# In[ ]:





# In[9]:


def semantic_search_FAISS(query, k = 5):
    query_embedding = get_embeddings([query]).numpy()
    
    
    desc_scores, desc_results = embeddings_dataset.get_nearest_examples("desc_embeddings", query_embedding, 100)
    
    desc_df = pd.DataFrame.from_dict(desc_results)
    desc_df["desc_scores"] = desc_scores
    
    name_scores, name_results = embeddings_dataset.get_nearest_examples("name_embeddings", query_embedding, 100)
    
    name_df = pd.DataFrame.from_dict(name_results)
    name_df["name_scores"] = name_scores
    
    name_df = name_df[["id", "name_scores"]]
    
    comb_df = desc_df.merge(name_df, how = 'inner', on = 'id')
    
    #Can alter the weights instead of 50/50 to assess scores
    comb_df['scores'] = comb_df["name_scores"] + comb_df["desc_scores"]
    
    comb_df.sort_values("scores", ascending = False, inplace = True)
    
    return comb_df[:k]


# In[15]:


res = semantic_search_FAISS("Single bedroom studio in Capitol Hill. Free wifi and Kitchen use")

for _, x in res.iterrows():
    print(x["name"])
    print()
    print(x['description'])
    print(x['scores'] / 100)
    print("-" * 80)


# In[16]:


res = semantic_search_FAISS("Wonderful located studio in cute Capitol Hill neighborhood. Spacious studio with wireless internet, monitor to plug in computer, tv, and a newly remodeled kitchen and bathroom")

for _, x in res.iterrows():
    print(x["name"])
    print()
    print(x['description'])
    print(x['scores'] / 100)
    print("-" * 80)


# In[ ]:


#Trying to figure out cosine similarity for TensorFlows
# embeddings_dataset[0]["desc_embeddings"]


# In[ ]:


# embeddings_dataset[1]["desc_embeddings"]


# In[ ]:


# cosine_sim(embeddings_dataset[0]["desc_embeddings"], embeddings_dataset[1]["desc_embeddings"])


# In[12]:


def semantic_search_CS(query, k = 5, alpha = 0.5):
    query_embedding = get_embeddings([query]).numpy()[0]
    
    cosine_scores = []
    
    for row in embeddings_dataset:
#         print(i["id"])
        name_score = cosine_sim(query_embedding, row["name_embeddings"])
        desc_score = cosine_sim(query_embedding, row["desc_embeddings"])
        cosine_scores.append([row["id"], alpha * name_score + (1-alpha) * desc_score])
    cosine_scores.sort(reverse = True, key = lambda x: x[1])
    return cosine_scores[:k]


# In[13]:


res_lst = semantic_search_CS("Capitol Hill Studio, free wifi and laundry")

for lst in res_lst:
    row = data_listing[data_listing["id"] == lst[0]].to_dict()
    print(f'SCORE: {lst[1]}')
    print(row["name"])
    print()
    
    print(row["description"])
    print("-" * 80)


# In[14]:


res_lst = semantic_search_CS("Wonderful located studio in cute Capitol Hill neighborhood. Spacious studio with wireless internet, monitor to plug in computer, tv, and a newly remodeled kitchen and bathroom")

for lst in res_lst:
    row = data_listing[data_listing["id"] == lst[0]].to_dict()
    print(f'SCORE: {lst[1]}')
    print(row["name"])
    print()
    
    print(row["description"])
    print("-" * 80)


# In[ ]:





# In[ ]:


# question = "Single bedroom studio in Capitol Hill. Free wifi and Kitchen use"
# question_embedding = get_embeddings([question]).numpy()
# question_embedding.shape


# In[ ]:


# len(embeddings_dataset)


# In[ ]:


# desc_scores, desc_samples = embeddings_dataset.get_nearest_examples(
#     "desc_embeddings", question_embedding, k = 100
# )


# name_scores, name_samples = embeddings_dataset.get_nearest_examples(
#     "name_embeddings", question_embedding, k = 100
# )

# name_samples_df = pd.DataFrame.from_dict(name_samples)
# name_samples_df["name_scores"] = name_scores
# # name_samples_df.sort_values("scores", ascending=False, inplace=True)
# name_samples_df = name_samples_df[["id", "name_scores"]]




# desc_samples_df = pd.DataFrame.from_dict(desc_samples)
# desc_samples_df["desc_scores"] = desc_scores
# desc_samples_df.sort_values("desc_scores", ascending=False, inplace=True)

# # for _, x in samples_df.iterrows():
# #     print(x["name"])
# #     print(x['description'])
# #     print(x["scores"])
# #     print("-" * 80)
# #     if _ == 10:
# #         break


# In[ ]:


# # pd.join()
# name_samples_df.shape

# # desc_samples_df.shape


# In[ ]:


# name_samples_df.shape[0] == desc_samples_df.shape[0]

# comb_samples = desc_samples_df.merge(name_samples_df, how = "inner", on = "id")


# In[ ]:


# comb_samples["total_score"] = comb_samples["desc_scores"] + comb_samples["name_scores"]

# comb_samples.sort_values("total_score", ascending=False, inplace=True)


# In[ ]:


# print("-" * 80)
# print(question)
# print("-" * 80)

# for _, x in comb_samples.iterrows():
#     print(x["name"])
#     print(x['description'])
#     print(x['total_score'])
#     print("-" * 80)
    
#     if _ == 9:
#         break


# In[ ]:


# scores, samples = embeddings_dataset.get_nearest_examples(
#     "name_embeddings", question_embedding, k=5
# )

# samples_df = pd.DataFrame.from_dict(samples)
# samples_df["scores"] = scores
# samples_df.sort_values("scores", ascending=False, inplace=True)
# print("-" * 80)
# print(question)
# print("-" * 80)
# for _, x in samples_df.iterrows():
#     print(x["name"])
#     print(x['description'])
#     print(x['scores'])
#     print("-" * 80)


# In[ ]:


#Work on figuring out how to embed the title as well

