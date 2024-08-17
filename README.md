# Towards Faster Deep Graph Clustering via Efficient Graph Auto-encoder

An official code for paper "[Towards Faster Deep Graph Clustering via Efficient Graph Auto-Encoder](https://dl.acm.org/doi/10.1145/3674983)". This repository is based on the unified framework for deep attribute graph clustering (https://github.com/Marigoldwu/A-Unified-Framework-for-Deep-Attribute-Graph-Clustering) .

## Requirements

The proposed FastDGC is implemented with python 3.10.12 on a Tesla T4 GPU.

Python package information is summarized in **requirements.txt**:

```
matplotlib==3.5.3
munkres==1.1.4
numpy==1.21.5
scikit_learn==1.0.2
torch==2.3.0
```

## Quick Start

Pre-training for new dataset

```shell
python main.py -P -M pretrain_egae_for_FastDGC -D cora -LS 1 -S 325
```

> - The pre-training code file is located at ./model/pretrain_egae_for_FastDGC/train.py.
> - Pre-trained datasets [acm, dblp, cora, cite, amap, wisc, texas], which are saved at ./pretrain/pretrain_egae/FastDGC/.

### Fine-tuning for pre-trained dataset

```shell
python main.py -M FastDGC -D acm -LS 10
```

>- You can modify the data set through the -D command. Optional data sets are [acm, dblp]. If you want to run other datasets, please go to the [website ](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering) to download.
>- You can use -TS to obtain the t-SNE visualization of clustering results, and the picture will be saved to the 'img/clustering/FastDGC'. Note that **LS** should be set to **1**, otherwise a picture will be drawn for each experiment
>- You can use -H to obtain the heatmap of embeddings, and the picture will be saved to the 'img/heatmap/FastDGC'. Note that **LS** should be set to **1**, otherwise a picture will be drawn for each experiment

## Colab Notebook

Or you can download and unzip the code to Google Drive, name it FastDGC, and then run it in Colab. [https://colab.research.google.com/drive/1u89965L3AejZVWVlE1B70MfqNESdtvtj?usp=sharing](https://colab.research.google.com/drive/1u89965L3AejZVWVlE1B70MfqNESdtvtj?usp=sharing)

Our paper is just accepted by ACM Transactions on Knowledge Discovery from Data. 

## Citation

If you use our code, please cite our paper as:

```
@article{ding2024towards,
author = {Ding, Shifei and Wu, Benyu and Ding, Ling and Xu, Xiao and Guo, Lili and Liao, Hongmei and Wu, Xindong},
title = {Towards Faster Deep Graph Clustering via Efficient Graph Auto-Encoder},
year = {2024},
issue_date = {September 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {18},
number = {8},
issn = {1556-4681},
url = {https://doi.org/10.1145/3674983},
doi = {10.1145/3674983},
journal = {ACM Trans. Knowl. Discov. Data},
month = {aug},
articleno = {202},
numpages = {23},
keywords = {Deep graph clustering, graph auto-encoder, graph neural networks, unsupervised learning}
}
```

