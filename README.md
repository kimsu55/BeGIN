# Delving into Instance-Dependent Label Noise in Graph Data: A Comprehensive Study and Benchmark 

This is the official repository for BeGIN benchmark. 


### What is it?


In the paper we introduce 10 graph datasets with various types of noisy labels: Cora-ML, WikiCS, Product-s, Children, History, Photo, Cornell, Texas, Washington, and Wisconsin.  You can download these datasets using `dataset`. Note: the datasets are stored as undirected, that is, each edge is stored only once. If you load the edges i


nto DGL or PyG, which treat all graphs as directed, do not forget to call `dgl.to_bidirected` or `pyg.transforms.ToUndirected` to double the edges.

Roman-empire is a word dependency graph based on the Roman Empire article from the [English Wikipedia](https://huggingface.co/datasets/wikipedia).

Amazon-ratings is a product co-purchasing network based on data from [SNAP Datasets](https://snap.stanford.edu/data/amazon-meta.html).

Minesweeper is a synthetic graph emulating the eponymous game.

Tolokers is a crowdsourcing platform workers network based on [data](https://github.com/Toloka/TolokerGraph) provided by [Toloka](https://toloka.ai).

Questions is an interaction graph of users of a question-answering website based on data provided by [Yandex Q](https://yandex.ru/q).

Our datasets come from different domains and exhibit a wide range of structual properties. We provide some statistics of our datasets in the table below: