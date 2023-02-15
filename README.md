# GENN, Graph-Enforced Neural Network for Attributed Graph Clustering

This repository contains the implementation of GENN. 


## Requirements

To install the requirements:

```setup
pip install -r requirements.txt
```


## Running

To run GENN-Augmented on Cora:

```
python train.py
```

To run GENN on Cora:

```
python train.py --augmented 0 --lr 5e-3
```



## Results

  Evaluation results on the attributed graph clustering task:

<img src="./cluster_performance.png" style="zoom:80%;" />
