# Deep_Learning_Notes
## Methods which performance well in real world regression issues

```

import numpy as np
from gudhi import RipsComplex, SimplexTree

# Generate a random distance matrix
dist_matrix = np.random.random((100, 100))
dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Symmetrize the matrix

# Create a Rips complex
rips_complex = RipsComplex(distance_matrix=dist_matrix, max_edge_length=np.max(dist_matrix))

# Create a simplex tree
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

# Compute persistence
simplex_tree.persistence()

# Use the first persistence diagram
d = np.array(simplex_tree.persistence_intervals_in_dimension(0))

# Calculate the sum of lifetimes
sum_of_lifetimes = (d[:,1] - d[:,0]).sum()

print(sum_of_lifetimes)

import numpy as np
import dionysus as d

# Generate a random distance matrix
dist_matrix = np.random.random((100, 100))
dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Symmetrize the matrix

# Convert the distance matrix to a list of lists
distance_list = dist_matrix.tolist()

# Create a filtration of Rips complexes using the distance matrix
filtration = d.fill_rips(distance_list, 2, np.max(dist_matrix))

# Compute persistence
m = d.homology_persistence(filtration)
diagrams = d.init_diagrams(m, filtration)

# Use the first persistence diagram
dgm = diagrams[0]
d = np.array([(pt.birth, pt.death) for pt in dgm])

# Calculate the sum of lifetimes
sum_of_lifetimes = (d[:,1] - d[:,0]).sum()

print(sum_of_lifetimes)



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
