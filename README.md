# DSC180Q1Project
DSC180 Project. Goal is to create a search algorithm and train it on the CodeSearchNet corpus. 

### Authors
Megan Huynh (mlhuynh@ucsd.edu)
William Hu (wyhu@ucsd.edu)

# Introduction

The goal of this project is to explore and build a search engine that can deliver good quantitative and qualitative results to code queries. We're seeking to obtain an NDCG score of over 0.2 and obtain results we deem qualitatively accurate.

### Architecture/Steps:
First our search engine will take in a natural language query and K desired results the user wants returned, tokenize it in both the semantic search embeddings space and the nlp tokenization vector, then it will go through various
matching algorithms (Cosine similarity, FAISS, ANNOY) etc and we will create a method of combining both the semantic results with the tf-idf/bm25 results and return the best K results.

### Algorithms
We plan to have two main components: the semantic search portion and the term-frequency/keyword matching portion.

For the semantic search, we want to embed a variety of fields (the function documentation, the query, the function code) with different trained models. Things we want to look into specifically are pre-trained code models such as codeBERT and training our own models. 

For the term-frequency/keyword portion we want to explore the tokenization of various fields similar to the semantic search. We want to look into n-grams, tokenizing the code strings, pre-processing the strings, etc. 

When grading the semantic and term frequency embeddings we plan to test different similarity functions such as: cosine similarity, euclidean distance, FAISS, ANNOY. 

After these portions are calculated, we want to combine them in an optimal manner. We want to test a variety of combinations such as linear combinations, first term-frequency then semantic, taking the top tf-idf then filling with semantic, and figuring out the best
performing hyper parameters for this. 

### Dataset
We are using the CodeSearchNet corpus, specifically the ~2M entries of functions including documentation. There is also 2 csvs; one containing 99 basic queries to test and one containing expert annotations of the given queries.

### Evaluation
Quantitative: We want to assess our results with the expertAnnotations on a variety of metrics including NDCG, MRR, and accuracy. We aim to match and surpass the NDCG baseline with ElasticSearch of 0.337.

Qualitative: Ultimately, we want results that pass the "eye test" so if given a query, are the results useful and we'll run queries and see if the results make sense.

### Improvements
Some goals we have for this project: Be able to use our algorithm to answer real life coding queries we have (LOL), developing models that are trained and return results in a reasonable time frame (don't want minute long waits), and match a top paper which 
has also done research on this very problem.

# How this Repo is Constructed (Currently)

## Folders

Dataset/Testing Folder: This is where the query and annotations csv files are for testing our model results.

csv_output Folder: This is where csv output files are put into when we're predicting results of queries.

WH Folder: This is where all of William Hu's older code files are. They're now put there as prior research. 

## Python Files & Jupyter Notebooks

relevanceeval.py: This file was taken from the CodeSearchNet repository (Link: https://github.com/github/CodeSearchNet) and is used to measure the NDCG score of our predictions

test.ipynb: This file is the "testing" grounds of our project so far. Has most of the code where the model is trained.

generate_results.ipynb: This file is where the output of our csv is compared to the results of the annotations.csv file.

# Setup Instructions

## Local

If you plan to run this project locally, you need to have an Nvidia graphics card with CUDA support. 

You need to install the Nvidia CUDA toolkit, find the proper CUDA version and download the corresponding PyTorch installation.

Then you can run anaconda and clone the DSC180_conda_env.txt file to get the code environment to run these files locally.

## DSMLP

For DSMLP machines, they should have the prerequisite packages installed so clone the repo after remoting onto your account and you can run any of the jupyter notebooks. 

## Accessing Data

This project utilizes the CodeSearchNet dataset. To access it, specifically the 2M annotated rows of data, we utilize HuggingFace. After importing HuggingFaces' datasets package, run ```load_dataset("code_search_net", "all")```. 

#### Disclaimer when loading data on DSMLP
I advise caution when running ```load_dataset("code_search_net", "all")``` as my datahub instance crashed and I had to clear the cache before being able to access it again. I recommend loading a smaller subset, possibly for just all the Python functions run 
```load_dataset("code_search_net", "python")```. This will load a smaller dataset and hopefully not crash your datahub.

*** 
Within the Dataset folder, there is also a Testing folder which contains two files: annotationStore.csv and queries.csv

The annotationStore file consists of ~4,000 rows of data in the format of query, github url, coding language, and expert annotation ranking. These annotation rankings vary from 1-10 in terms of relevance and will be utilized to calculate the NDCG and MRR scores. 

The queries file consists of 99 basic coding queries which are the basis of the testing. 

## Storing the Data

In the test.ipnyb Jupyter Notebook file, there are cells which specifically save and load pickled objects to reduce time spent training the large models. It's recommended to create a folder within the /Dataset folder (Example: pickleObjects/), include it in the gitignore file, and accessing the saved models/results there. 

# Code Credit

We utilized pre-existing code from many people. Specifically code from: the CodeSearchNet (https://github.com/github/CodeSearchNet), HuggingFace tutorials (https://huggingface.co/learn/nlp-course/chapter5/6?fw=pt) and William Scott (https://github.com/williamscott701/Information-Retrieval/tree/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score)

https://medium.com/@evertongomede/understanding-the-bm25-ranking-algorithm-19f6d45c6ce Was instrumental in our own implementation of BM25.

https://medium.com/@readsumant/understanding-ndcg-as-a-metric-for-your-recomendation-system-5cd012fb3397 Was also helpful for our implmentation of NDCG.

There are comments above specific functions/code cells/python files crediting the respective authors when using their code. 

### Disclaimer:

Much of this code is not standardized yet and is in the "experimental" stage and is done in Jupyter Notebook files. We have plans to eventually turn them all into separate python scripts (one to train the model, one to automatically generate the results, one to compare the results to the annotations, etc)

