# Towards Faster Deep Graph Clustering via Efficient Graph Auto-encoder

An official code for paper "Towards Faster Deep Graph Clustering via Efficient Graph Auto-Encoder". This repository is based on the unified framework for deep attribute graph clustering (https://github.com/Marigoldwu/A-Unified-Framework-for-Deep-Attribute-Graph-Clustering) .

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

```
python main.py -M FastDGC -D acm -LS 10
```

>- You can modify the data set through the -D command. Optional data sets are [acm, dblp]. If you want to run other datasets, please go to the [website ](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering) to download.
>- You can use -TS to obtain the t-SNE visualization of clustering results, and the picture will be saved to the 'img/clustering/FastDGC'. Note that **LS** should be set to **1**, otherwise a picture will be drawn for each experiment
>- You can use -H to obtain the heatmap of embeddings, and the picture will be saved to the 'img/heatmap/FastDGC'. Note that **LS** should be set to **1**, otherwise a picture will be drawn for each experiment

## Colab Notebook

Or you can download and unzip the code to Google Drive, name it FastDGC, and then run it in Colab. [https://colab.research.google.com/drive/1u89965L3AejZVWVlE1B70MfqNESdtvtj?usp=sharing](https://colab.research.google.com/drive/1u89965L3AejZVWVlE1B70MfqNESdtvtj?usp=sharing)

Our paper is just accepted by ACM Transactions on Knowledge Discovery from Data. 