# Delving into Instance-Dependent Label Noise in Graph Data: A Comprehensive Study and Benchmark 

This is the official repository for BeGIN benchmark. 


## What is it?

**BeGIN** offers 10 graph datasets with six different noise types, along with evaluation benchmark for
various noise-handling strategies, including various GNN architectures, noisy label detection, and noise robust learning.
To simulate more realistic noise beyond class-dependent assumptions, we first introduce various types of instance-dependent label noise.


| Method | Class-dependent | Instance-dependent |
|--------|----------------|--------------------|
|        |                | topology | feature |
| Type A | ✅             | ✅        | ❌      |
| Type B | ❌             | ✅        | ✅      |
| Type C | ✅             | ❌        | ✅      |


| Label noise types     | Class-dependent | Topology-dependent | Feature-dependent |
| --------------------- | --------------- | ------------------ | ----------------- |
| Uniform               | <p align="center">✔</p>|     | |
| Pairwise              | <p align="center">✔</p>|     | | 
| Feature               |  <p align="center">✔</p>     |  | <p align="center">✔</p>|
| Topology              | <p align="center">✔</p>|  | <p align="center">✔</p>|
| Confidence            |<p align="center">✔</p>| <p align="center">✔</p>|<p align="center">✔</p>|
| LLM                   |   | |<p align="center">✔</p>|


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

### Creat a dataset with LLM-based label noise
noisy_dataset = NoisyDataset(root='./data', name='cornell')

### Generate other types of label noise
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