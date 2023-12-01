# %%
import pandas as pd
import numpy as np
import relevanceeval


# # Goal
# This jupyter notebook serves as an area to test how to automatically test the results of our trained model to the expert annotations. It serves as a testing ground. 

# %%
# Loading in the queries csv file.
def load_csv(filepath):
    # queries = pd.read_csv("./Dataset/Testing/queries.csv")
    return pd.read_csv(filepath)



def print_results(languages_predicted, languages_rs, relevance_scores, predictions):
    print('% of URLs in predictions that exist in the annotation dataset:')
    for language in languages_predicted:
        if language in languages_rs:
            print(f'\t{language}: {relevanceeval.coverage_per_language(predictions[language], relevance_scores[language])*100:.2f}%')

    print('% of URLs in predictions that exist in the annotation dataset (avg relevance > 0):')
    for language in languages_predicted:
        if language in languages_rs:
            print(f'\t{language}: {relevanceeval.coverage_per_language(predictions[language], relevance_scores[language], with_positive_relevance=True) * 100:.2f}%')

    print('NDCG:')
    for language in languages_predicted:
        if language in languages_rs:
            print(f'\t{language}: {relevanceeval.ndcg(predictions[language], relevance_scores[language]):.3f}')


# %%


# %%
# Results for baseline 20k NO FILTERING ANSWERS
# % of URLs in predictions that exist in the annotation dataset:
# 	go: 0.00%
# 	java: 0.25%
# 	javascript: 0.00%
# 	php: 0.32%
# 	python: 0.20%
# 	ruby: 0.64%
# % of URLs in predictions that exist in the annotation dataset (avg relevance > 0):
# 	go: 0.00%
# 	java: 0.23%
# 	javascript: 0.00%
# 	php: 0.47%
# 	python: 0.23%
# 	ruby: 0.00%
# NDCG:
# 	go: 0.000
# 	java: 0.006
# 	javascript: 0.000
# 	php: 0.001
# 	python: 0.005
# 	ruby: 0.000

# %%
#Newer working area

def create_lj_answers_OLD(filepathPred, filePathAnswer, temp_output):
    annotationStore = load_csv(filePathAnswer)
    predictions = load_csv(filepathPred)

    res_annot_merged = pd.merge(predictions, annotationStore, how="left", left_on="url", right_on="GitHubUrl")
    ans_in_res_DF = res_annot_merged[res_annot_merged["GitHubUrl"].isnull() == False][["Language","Query", "GitHubUrl", "Relevance", "Notes"]]
    ans_in_res_DF.to_csv(temp_output) #./csv_output/rel_score_combined.csv


    relevance_scores = relevanceeval.load_relevances(temp_output)
    predictions = relevanceeval.load_predictions(filepathPred)
    languages_predicted = sorted(set(predictions.keys()))
    languages_rs = sorted(set(relevance_scores.keys()))

    print_results(languages_predicted, languages_rs, relevance_scores, predictions)


def create_lj_answers_NEW(filepathPred, filePathAnswer):
    annotationStore = load_csv(filePathAnswer)
    predictions = load_csv(filepathPred)

    
    res_annot_merged = pd.merge(predictions, annotationStore, how="left", left_on="url", right_on="GitHubUrl")
    res_annot_merged["Relevance"].fillna(0, inplace=True)
    ans_in_res_DF = res_annot_merged[res_annot_merged["GitHubUrl"].isnull() == False][["Language","Query", "GitHubUrl", "Relevance", "Notes"]]
    # ans_in_res_DF.to_csv(temp_output) #./csv_output/rel_score_combined.csv
    
    queries_with_annotations = list(ans_in_res_DF["Query"].unique())
    ndcgs = []
    for qwa in queries_with_annotations:
        df_query = res_annot_merged[res_annot_merged["query"] == qwa]
        rel_lst = df_query["Relevance"].to_list()

        dcg = 0
        for i, val in enumerate(rel_lst):
            dcg += (val) / (np.log2(i + 2))

        # print(dcg)
        idcg = 0
        for i, val in enumerate(sorted(rel_lst, reverse=True)):
            idcg += (val) / (np.log2(i + 2))
        # print(idcg)
        if idcg == 0:
            ndcgs.append(0)
            print(f"For query: {qwa}, no relevant results")
        else: ndcgs.append(dcg / idcg)
    print(queries_with_annotations, len(queries_with_annotations))
    print(np.mean(ndcgs))