import steelthread

if __name__ == '__main__':
    model = steelthread.create_model(100000, 1, True, False, "query_results_lc_naive_custom", "BM25", 0.75, True)
    steelthread.create_results(model, 50)