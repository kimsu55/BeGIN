# Delving into Instance-Dependent Label Noise in Graph Data: A Comprehensive Study and Benchmark 

This is the official repository for BeGIN benchmark. 


### What is it?

In the paper we introduce 10 graph datasets with various types of noisy labels: Cora-ML, WikiCS, Product-s, Children, History, Photo, Cornell, Texas, Washington, and Wisconsin.  You can download these datasets using `dataset`. Note: the datasets are stored as undirected, that is, each edge is stored only once. If you load the edges i



### Required Dependencies?
- Python 3.8+
- torch>=2.1.0
- pyg>=2.5.0
- huggingface-hub
- pandas
- nltk