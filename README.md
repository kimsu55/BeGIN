# Delving into Instance-Dependent Label Noise in Graph Data: A Comprehensive Study and Benchmark 

This is the official repository for BeGIN benchmark. 


## What is it?

**BeGIN** offers 10 graph datasets with six different noise types, along with evaluation benchmark for
various noise-handling strategies, including various GNN architectures, noisy label detection, and noise robust learning.
To simulate more realistic noise beyond class-dependent assumptions, we first introduce various types of instance-dependent label noise such as Feature, Topology, Confidence and LLM based label noise.



#### Label Noise Types
| Label noise types     | Class-dependent | Instance-dependent (Topology) | Instance-dependent (Feature) |
| --------------------- | --------------- | ------------------ | ----------------- |
| Uniform               | <p align="center">✔</p>|     | |
| Pairwise              | <p align="center">✔</p>|     | | 
| Feature               |  <p align="center">✔</p>     |  | <p align="center">✔</p>|
| Topology              | <p align="center">✔</p>|  <p align="center">✔</p>| |
| Confidence            |<p align="center">✔</p>| <p align="center">✔</p>|<p align="center">✔</p>|
| LLM                   |   | |<p align="center">✔</p>|


#### Built-in Datasets

This framework allows users to use real-world datasets as follows:
  | Category    | Cora-ML | WikiCS | Product-s | Children | History | Photo | Cornell | Texas | Washington | Wisconsin |
|------------|---------|--------|-----------|----------|---------|-------|---------|-------|------------|-----------|
| **# Nodes** | 2,995   | 11,701  | 54,025   | 76,875   | 41,551  | 48,362 | 191     | 187   | 229        | 265       |
| **# Edges** | 8,158   | 216,123 | 74,420   | 1,554,578 | 358,574 | 500,939 | 292     | 310   | 394        | 510       |





##  Quick Example for BeGIN Dataset 


#### Required Dependencies?
- Python 3.8+
- torch>=2.1.0
- pyg>=2.5.0
- huggingface-hub
- pandas
- nltk




```python
from dataset.BeGINdataset import NoisyDataset
from dataset.noisify import noisify_dataset

### Creat a dataset with LLM-based label noise
noisy_dataset = NoisyDataset(root='./data', name='cornell')

### Generate other types of label noise
noisy_labels, transition_matrix = noisify_dataset(noisy_dataset, noise_type='topology')

```

