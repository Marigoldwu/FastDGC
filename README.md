# Towards Faster Deep Graph Clustering via Efficient Graph Auto-encoder

An official code for paper "Towards Faster Deep Graph Clustering via Efficient Graph Auto-Encoder". This repository is based on the unified framework for deep attribute graph clustering (https://github.com/Marigoldwu/A-Unified-Framework-for-Deep-Attribute-Graph-Clustering) .

## Requirements

The proposed GCMA is implemented with python 3.7 on a Tesla T4 GPU.

Python package information is summarized in **requirements.txt**:

```
matplotlib==3.5.3
munkres==1.1.4
numpy==1.21.5
scikit_learn==1.0.2
torch==1.11.0
```

## Quick Start

```
python main.py -M FastDGC -D acm -LS 10 -DS FastDGC在ACM数据集上进行10次实验
```

>- You can use -TS to obtain the t-SNE visualization of clustering results, and the picture will be saved to the 'img/clustering/FastDGC'.
>- You can use -H to obtain the heatmap of embeddings, and the picture will be saved to the 'img/heatmap/FastDGC'.
>- Note that **LS** should be set to **1**, otherwise a picture will be drawn for each experiment

Our paper is under review. More details will be released after the paper is accepted.