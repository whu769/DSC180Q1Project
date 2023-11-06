# DSC180Q1Project
DSC180 Project. Goal is to create a search algorithm and train it on the CodeSearchNet corpus. 


# Setup Instructions

## Local

If you plan to run this project locally, you need to have an Nvidia graphics card with CUDA support. 

You need to install the Nvidia CUDA toolkit, find the proper CUDA version and download the corresponding PyTorch installation.

Then you can run anaconda and clone the DSC180_conda_env.txt file to get the code environment to run these files locally.

## DSMLP

For DSMLP machines, they should have the prerequisite packages installed so clone the repo after remoting onto your account and you can run any of the jupyter notebooks. 

## Accessing Data

This project utilizes the CodeSearchNet dataset. To access it, specifically the 2M annotated rows of data, we utilize HuggingFace. After importing HuggingFaces' datasets package, run ```load_dataset("code_search_net", "all")```. 

Within the Dataset folder, there is also a Testing folder which contains two files: annotationStore.csv and queries.csv

The annotationStore file consists of ~4,000 rows of data in the format of query, github url, coding language, and expert annotation ranking. These annotation rankings vary from 1-10 in terms of relevance and will be utilized to calculate the NDCG and MRR scores. 

The queries file consists of 99 basic coding queries which are the basis of the testing. 

## Storing the Data

In the test.ipnyb Jupyter Notebook file, there are cells which specifically save and load pickled objects to reduce time spent training the large models. It's recommended to create a folder within the /Dataset folder (Example: pickleObjects/), include it in the gitignore file, and accessing the saved models/results there. 

### Disclaimer:

Much of this code is not standardized yet and is in the "experimental" stage and is done in Jupyter Notebook files. We have plans to eventually turn them all into separate python scripts (one to train the model, one to automatically generate the results, one to compare the results to the annotations, etc)

