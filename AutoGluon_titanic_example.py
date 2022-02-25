from autogluon.tabular import TabularDataset, TabularPredictor
#training
trian_data = TabularDataset('train.csv')
id, label = 'PassengerId', 'Survived'
predictor = TabularPredictor(label = label).fit(train_data.drop(columns = [id]))

import pandas as pd
test_data = TabularDataset('test.csv')
preds = predictor.predict(test_data.drop(columns = [id]))
