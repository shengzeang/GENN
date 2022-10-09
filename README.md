# GENN, Graph-Enforced Neural Networks for Attributed Graph Clustering

This repository is the official implementation of GENN. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train GENN-Augmented:

```
python train.py --dw True --decoupled False
```

To train GENN:

```
python train.py --dw False --decoupled False
```

To train GENN-Simplified:

```
python train.py --dw False --decoupled True
```




## Results

  Evaluation results on the clustering task:

<img src="./cluster_performance.png" style="zoom:80%;" />

Evaluation results on the link prediction task:

<img src="./link_performance.png" style="zoom:80%;" />