# DSC180Q1Project
DSC180 Project. The goal is to create a search algorithm and train it on the CodeSearchNet corpus. 

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

For the semantic search, we want to embed a variety of fields (the function documentation, the query, the function code) with different trained models. Things we want to look into specifically are pre-trained code models such as codeBERT and sBert. 

For the term-frequency/keyword portion we want to explore the tokenization of various fields similar to the semantic search. We want to look into n-grams, tokenizing the code strings, pre-processing the strings, etc. 

When grading the semantic and term frequency embeddings we plan to test different similarity functions such as: cosine similarity, euclidean distance, FAISS and other approximate nearest neighbor algorithms. 

After these portions are calculated, we want to combine them in an optimal manner. We want to test a variety of combinations such as linear combinations, first term-frequency then semantic, taking the top tf-idf then filling with semantic.

### Dataset
We are using the CodeSearchNet corpus, specifically the ~2M entries of functions including documentation. There is also 2 csvs; one containing 99 basic queries to test and one containing expert annotations of the given queries.

### Evaluation
Quantitative: We want to assess our results with the expertAnnotations on a variety of metrics including NDCG, MRR, and accuracy. We aim to reach an NDCG score of 0.200 or above.

Qualitative: Ultimately, we want results that pass the "eye test" so if given a query, are the results useful and we'll run queries and see if the results make sense.

### Improvements
Some goals we have for this project: Be able to use our algorithm to answer real life coding queries we have, developing models that are trained and return results in a reasonable time frame (don't want minute long waits), and match a top paper which has also done research on this very problem.

# How this Repo is Constructed 

## Folders

Dataset/Testing Folder: This is where the query and annotations csv files are for testing our model results.

csv_output Folder: This is where csv output files are put into when we're predicting results of queries.

Research: This is where all of our various JupyterNotebooks, python files and older code from individual assignments (WH and MH for our respective works) are put into.

## Python Files & Jupyter Notebooks

gen_results.py: This file is used to measure the NDCG score of our predictions. It matches the predictions with the expert annotations that are available. The NDCG results
are outputted onto console.

run.py: This file is the file that users will run.

searchmodel.py: This file is the searchmodel instance. It has the parameters for the model and is has the methods that initializes/loads the inverted index and embeddings data that is used
to query.

steelthread.py: This file is the "backbone" of the file. It is what the run.py file calls and builds the searchmodel instance to run the 99 queries.

final_test.ipynb: This jupyter notebook file is 

# Setup Instructions

## Local

If you plan to run this project locally, you need to have an Nvidia graphics card with CUDA support. 

You need to install the Nvidia CUDA toolkit, find the proper CUDA version and download the corresponding PyTorch installation.

Then you can run anaconda and clone the environment.yml file to get the code environment to run these files locally.

## DSMLP

For DSMLP machines, they should have the majority of the prerequisite packages installed so clone the repo after remoting onto your account and you can run any of the jupyter notebooks. 
You will need to download the CodeSearchNet data from HuggingFace though. 

Things to do once cloning the git:
* For local machines, clone the Anaconda environment with the "environment.yml"
* For DSMLP, initialize an instance from the DSC180A workspace as it has alot of prerequisite ML packages installed (We recommend something like this: launch-scipy-ml.sh -c 8 -m 32 -g 1 -W DSC180A_FA23_A00)
* Make sure to run the Jupyter Notebook file inside the Dataset Folder
* Create a folder in the repo called "pickleObjects" - This is where all of the trained inverted indexes and embeddings will be stored.
* Make sure nltk is installed with the nltk english stopwords. (nltk.download('stopwords'))


## Accessing Data

Within the Dataset folder will be a Jupyter Notebook file that will have cells that will allow you to choose the specific CodeSearchDataset. It will download and save it in a CodeSearchCorpus/ folder within the Dataset folder. 

This project utilizes the CodeSearchNet dataset. To access it, specifically the 2M annotated rows of data, we utilize HuggingFace. After importing HuggingFaces' datasets package, run ```load_dataset("code_search_net", "all")```. 

#### Disclaimer when loading data on DSMLP
I advise caution when running ```load_dataset("code_search_net", "all")``` as my datahub instance crashed and I had to clear the cache before being able to access it again. I recommend loading a smaller subset, possibly for just all the Python functions run 
```load_dataset("code_search_net", "python")```. This will load a smaller dataset and hopefully not crash your datahub (I've crashed mine a couple times now).

*** 
Within the Dataset folder, there is also a Testing folder which contains two files: annotationStore.csv, annotationStore_UNIQUE.csv and queries.csv

The annotationStore file consists of ~4,000 rows of data in the format of query, github url, coding language, and expert annotation ranking. These annotation rankings vary from 1-10 in terms of relevance and will be utilized to calculate the NDCG and MRR scores. 

There are some repeats in annotationStore.csv however, and annotationStore_UNIQUE.csv is a dataset that eliminates duplicates. Only the first occurrence is kept (which is not the perfect method of doing this  so if you wanted to condense annotationStore a different way feel free to do so).

The queries file consists of 99 basic coding queries which are the basis of the testing. 

## Storing the Data

In the test.ipnyb Jupyter Notebook file, there are cells which specifically save and load pickled objects to reduce time spent training the large models. It's recommended to create a folder within the /Dataset folder (Example: pickleObjects/), include it in the gitignore file, and accessing the saved models/results there. 

# Code Credit

We utilized pre-existing code from many people. Specifically code from: 
* The CodeSearchNet (https://github.com/github/CodeSearchNet) 
* HuggingFace tutorials (https://huggingface.co/learn/nlp-course/chapter5/6?fw=pt)
* William Scott (https://github.com/williamscott701/Information-Retrieval/tree/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score)
* https://medium.com/@evertongomede/understanding-the-bm25-ranking-algorithm-19f6d45c6ce Was instrumental in our own implementation of BM25.
* https://medium.com/@readsumant/understanding-ndcg-as-a-metric-for-your-recomendation-system-5cd012fb3397 Was also helpful for our implmentation and understanding of NDCG.

There are comments above specific functions/code cells/python files crediting the respective authors when using their code. 

Most importantly, a HUGE thanks to our mentor Colin Jemmott for helping us with everything from optimizing code to suggestions to improving our project!

