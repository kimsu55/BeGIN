# Delving into Instance-Dependent Label Noise in Graph Data: A Comprehensive Study and Benchmark 

This is the official repository for BeGIN benchmark. 


## What is it?

**BeGIN** offers 
In the paper we introduce 10 graph datasets with various types of noisy labels: Cora-ML, WikiCS, Product-s, Children, History, Photo, Cornell, Texas, Washington, and Wisconsin.  You can download these datasets using `dataset`.


| Label noise types     | Class-dependent | Topology-dependent | Feature-dependent |
| --------------------- | --------------- | ------------------ | ----------------- |
| Uniform               | ✅
| Pairwise               | ✅
| Feature
| Topology
| Confidence
| LLM


## Required Dependencies?
- Python 3.8+
- torch>=2.1.0
- pyg>=2.5.0
- huggingface-hub
- pandas
- nltk



##  Quick Example

```python
from dataset.BeGINdataset import NoisyDataset
from dataset.noisify import noisify_dataset

### creat dataset with LLM-based label noise
noisy_dataset = NoisyDataset(root='./data', name='cornell')

### generate other types of label noise
noisy_labels, transition_matrix = noisify_dataset(noisy_dataset, noise_type='topology')

```

## Built-in Datasets

This framework allows users to use real-world datasets as follows:
  | Dataset                                                 | # Nodes | # Edges |
  | ------------------------------------------------------- | ------- | ------- |
  | [Cora-ML](https://github.com/kimiyoung/planetoid)       | 2,995   | 8,158   |
  | [WikiCS](https://github.com/kimiyoung/planetoid)        | 11,701  | 216,123  |
  | [Product-s](https://github.com/kimiyoung/planetoid)     | 54,025   | 74,420   |
  | [Children](https://openreview.net/forum?id=S1e2agrFvS)  | 76,875   | 1,554,578     |
  | [History](https://openreview.net/forum?id=S1e2agrFvS)   | 41,551    | 358,574    |
  | [Photo](https://openreview.net/forum?id=S1e2agrFvS)     | 48,362   | 500,939  |
  | [Cornell](https://openreview.net/forum?id=S1e2agrFvS)   | 191     | 292    |
  | [Texas](https://openreview.net/forum?id=S1e2agrFvS)     | 187   | 310  |
  | [Washington](https://openreview.net/forum?id=S1e2agrFvS) | 229     | 394     |
  | [Wisconsin](https://openreview.net/forum?id=S1e2agrFvS) | 265     | 510     |