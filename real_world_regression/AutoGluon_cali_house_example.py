from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
#training
trian_data = TabularDataset('train.csv')
id, label = 'Id', 'Sold Price'
#data wash
large_val_cols = ['lot','total interior livable area',...]
for i in large_val_cols + [label]:
  train_data[c] = np.log(train_data[c] + 1)


predictor = TabularPredictor(label = label).fit(
  train_data.drop(columns = [id]),
  hyperparameters = 'multimodal',
  num_stack_level=1,num_bag_folds = 5)

import pandas as pd
test_data = TabularDataset('test.csv')
preds = predictor.predict(test_data.drop(columns = [id]))
