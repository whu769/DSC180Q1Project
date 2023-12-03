"""Search Engine NDCG Analysis Script

This script takes in two files: a csv of predictions and a csv of
expert annotations (the answers)

This script requires that `pandas` and `numpy` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * create_lj_answers - Prints out the average NDCG score given the predictions.csv and answers.csv
"""
import pandas as pd
import numpy as np
import relevanceeval


def load_csv(filepath):
    """
    Function that loads in the csv given the filepath

    Parameters
    ----------
    filepath : str
        The filelocation of the csv file
    """
    
    return pd.read_csv(filepath)

def create_lj_answers_NEW(filePathPred, filePathAnswer):
    """
    Function that loads in the predictions and answers, and the
    prints out the average NDCG score.

    Parameters
    ----------
    filePathPred : str
        The file location of the predictions csv file
    filePathAnswer : str
        The file location of the answers csv file
    """
    annotationStore = load_csv(filePathAnswer)
    predictions = load_csv(filePathPred)

    
    res_annot_merged = pd.merge(predictions, annotationStore, how="left", left_on="url", right_on="GitHubUrl")
    res_annot_merged["Relevance"].fillna(0, inplace=True)
    ans_in_res_DF = res_annot_merged[res_annot_merged["GitHubUrl"].isnull() == False][["Language","Query", "GitHubUrl", "Relevance", "Notes"]]
   
    
    queries_with_annotations = list(ans_in_res_DF["Query"].unique())
    ndcgs = []
    empty_res = 0
    for qwa in queries_with_annotations:
        df_query = res_annot_merged[res_annot_merged["query"] == qwa]
        rel_lst = df_query["Relevance"].to_list()

        dcg = 0
        for i, val in enumerate(rel_lst):
            dcg += (val) / (np.log2(i + 2))

        
        idcg = 0
        for i, val in enumerate(sorted(rel_lst, reverse=True)):
            idcg += (val) / (np.log2(i + 2))
        
        if idcg == 0:
            ndcgs.append(0)
            empty_res += 1
            # print(f"For query: {qwa}, no relevant results")
        else: ndcgs.append(dcg / idcg)
    print(f"For the predictions, a total of {len(ndcgs)} queries were matched. \
          {empty_res} of the Queries had no relevant results")
    print(f"NDCG: {np.mean(ndcgs)}")