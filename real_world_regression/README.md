# Deep_Learning_Notes
## Methods which performance well in real world regression issues

```
import numpy as np
from gudhi import RipsComplex
from gudhi import SimplexTree

# Generate some random data points
data = np.random.random((100, 2))

# Create a Rips complex
rips_complex = RipsComplex(points=data, max_edge_length=2.0)

# Create a simplex tree
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

# Compute persistence
simplex_tree.persistence()

# Extract the persistence diagrams
persistence_diagrams = simplex_tree.persistence_intervals_in_dimension(0), simplex_tree.persistence_intervals_in_dimension(1)

# Do something with the persistence diagrams
print(persistence_diagrams)



import numpy as np
import dionysus as d

# Generate some random data points
data = np.random.random((100, 2))

# Create a filtration of Rips complexes
filtration = d.fill_rips(data, 2, 2.0)

# Compute persistence
m = d.homology_persistence(filtration)
diagrams = d.init_diagrams(m, filtration)

# Convert to a format similar to other libraries (list of arrays)
persistence_diagrams = [np.array([(pt.birth, pt.death) for pt in dgm]) for dgm in diagrams]

# Do something with the persistence diagrams
print(persistence_diagrams)


```

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
