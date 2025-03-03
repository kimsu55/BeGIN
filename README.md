# Delving into Instance-Dependent Label Noise in Graph Data: A Comprehensive Study and Benchmark 

This is the official repository for BeGIN benchmark. 


## What is it?

**BeGIN** provides **graph datasets with instance-dependent label noise**. 
Specifically it offers 10 graph datasets with six different noise types, along with evaluation benchmark for
various noise-handling strategies, including various GNN architectures, noisy label detection, and noise robust learning.
<!-- To simulate more realistic noise beyond class-dependent assumptions, we introduce various types of instance-dependent label noise such as Feature, Topology, Confidence and LLM based label noise. -->



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
|------------|:------:|:------:|:---------:|:--------:|:-------:|:-----:|:-------:|:-----:|:----------:|:---------:|
| **# Nodes** | 2,995   | 11,701  | 54,025   | 76,875   | 41,551  | 48,362 | 191     | 187   | 229        | 265       |
| **# Edges** | 8,158   | 216,123 | 74,420   | 1,554,578 | 358,574 | 500,939 | 292     | 310   | 394        | 510       |



##  Quick Example for BeGIN Dataset 
The ``NoisyGraphDataset`` class allows you to generate a graph dataset with LLM-based label noise for all datasets except 'Products', 'Cora-ML', and 'WikiCS', which have their own specific classes (``NoisyProducts``, ``NoisyCoraML``, and ``NoisyWikiCS``).

Additionally, you can create label noise from various noise types by setting the noise_type parameter to one of the following:
``uniform``,  ``pairwise``, ``feature``, ``topology``, ``confidence``, ``llm``.

#### Required Dependencies
- Python 3.8+
- torch>=2.1.0, torch_scatter
- pyg>=2.5.0
- huggingface-hub
- nltk
- numpy, scikit-learn, pandas, tqdm, ruamel.yaml


```python
from dataset.BeGINdataset import NoisyGraphDataset, NoisyProducts, NoisyCoraML, NoisyWikiCS
from dataset.noisify import noisify_dataset

### Creat a dataset with LLM-based label noise
noisy_dataset = NoisyGraphDataset(root='./data', name='cornell')

### Generate other types of label noise
noisy_label, transition_matrix = noisify_dataset(noisy_dataset, noise_type='topology')

```


##  Quick Example for Noisy label detector 
BeGIN provides a noisy label detector to help identify mislabeled data in graph datasets.
The following command trains a noisy label detection model on the Cora-ML dataset using Graph Convolutional Networks (GCN) with LLM-based label noise.
The ``data`` parameter must be one of the following: ``cora_ml``, ``wikics``,  ``products``, ``children``,``history``, ``photo``,  ``cornell``, ``texas``, ``washington``, ``wisconsin``.
The ``method`` parameter must be one of the following model architectures: ``gcn``,  ``gin``, ``gat``, ``sage``, ``mlp``.

```python

python train_detector.py  --data cora_ml --noise_type llm  --method gcn --device cuda
```


## Noisy Robust Learning  
BeGIN is designed to benchmark and advance learning under label noise in graph datasets. Our contribution includes:
-  A systematic evaluation framework for studying the impact of instance-dependent label noise on noise robust models.
- Pre-configured training pipelines that allow users to efficiently train and compare multiple noise-robust models.
- Extensive configuration files to enable flexible experimentation with different noise-handling techniques and architectures.

####  Quick Example using GNNs 
BeGIN provides a node classifier to train and evaluate graph neural networks (GNNs) on noisy datasets.
It enables users to analyze the impact of various label noise types on node classification performance.
The parameters are the same as those mentioned above.

```python

python train_nodeclassifier.py  --data cora_ml --noise_type llm  --method gcn --device cuda
```

#### Additional Models using NoisyGL 
To further strengthen our benchmark, we integrate NoisyGL ([NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise](https://arxiv.org/abs/2406.04299)) as a Git submodule. This incorporation expands BeGIN by providing access to a diverse suite of state-of-the-art noise-robust GNN models, enabling more comprehensive and scalable evaluations.


To ensure that NoisyGL is properly initialized before running experiments:
```bash
cd BeGIN
git submodule update --init --recursive
```
This command pulls the latest version of NoisyGL, allowing seamless integration with BeGIN for evaluating noise-robust GNN architectures.
