# Deep_Learning_Notes
## Methods which performance well in real world regression issues

[The link for Autogluon](https://github.com/awslabs/autogluon)
 ```
#Autogluon
python3 -m pip install -U pip
python3 -m pip install -U setuptools wheel
#for CPU 
python3 -m pip install -U "mxnet<2.0.0"
#for GPU
python3 -m pip install -U "mxnet_cu101<2.0.0"
python3 -m pip install autogluon
 ```
 [The link for h20](https://www.h2o.ai/products/h2o/)
 ```
#h2o
This is also a auto machine learning tools
 ```
 
 
 [The link for random forest](https://en.wikipedia.org/wiki/Random_forest)
 
```
#random forest(This method does not belong to deep learning but it works good for some regression issues)
pseudocode: 1. select random subsets from a dataset
            2. construct decision tree for every subsets
            3. let test data pass every decision tree, prediction with most votes will be the final answer
```
