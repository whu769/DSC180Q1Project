"""
run.py

The script to run on command line to create models and test the 99 queries.
Will print out NDCG results.
"""
import steelthread
import argparse

#Create argparse variables
parser = argparse.ArgumentParser()
parser.add_argument("--data_size", type = int, required= True, help="Number of rows to take from dataset")
parser.add_argument("--random_seed", type = int, required= True, help="Seed to randomly shuffle data")
parser.add_argument("--loadEmbed", type = bool, required= True, help= "Whether to create the embeddings or not")
parser.add_argument("--loadTF", type = bool, required= True, help= "Whether to create the inverted index or not")
parser.add_argument("--queryMethod", type = str, required= True, help= "Which method for the search engine to query.\
                    You can choose from: query_results_lc_naive_custom, query_results_tfidf, query_results_BM25, \
                    query_results_embed, query_results_lc_norm, query_results_odds_evens, query_results_faiss_kw \
                    query_results_faiss_cos")
parser.add_argument("--tfMethod", type = str, required= True, help= "Which method for the search engine to query.\
                    You can choose from: BM25 or TFIDF")
parser.add_argument("--tf_alpha", type = float, required= True, help= "Float value setting the Linear Combination Weight")
parser.add_argument("--bigrams", type = bool, required= True, help= "Boolean determining bigram support or not")
parser.add_argument("--res_per_query", type = int, required= True, help= "Int determining the k amount of top results to return per query.")

args = vars(parser.parse_args())

#takes in argparse variables and creates the corresponding model and prints out results
if __name__ == '__main__':
    # print(args)
    model = steelthread.create_model(args["data_size"], args["random_seed"], args["loadEmbed"], 
                                     args["loadTF"], args["queryMethod"], args["tfMethod"], args["tf_alpha"]
                                     , args["bigrams"])
    steelthread.create_results(model, args["res_per_query"])


# Here is an example command to run:
# python run.py --data_size 100000 --random_seed 1 --loadEmbed True --loadTF True --queryMethod "query_results_lc_naive_custom" --tfMethod "BM25" --tf_alpha 0.75 --bigrams True --res_per_query 50

