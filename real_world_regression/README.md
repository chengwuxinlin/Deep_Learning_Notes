# Deep_Learning_Notes
## Methods which performance well in real world regression issues



```
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

# Define the encoder
input_dim = 13000
latent_dim = 500

inputs = Input(shape=(input_dim,))
h = Dense(1024, activation='relu')(inputs)
h = Dense(512, activation='relu')(h)
encoded = Dense(latent_dim, activation='relu')(h)

# Define the decoder
decoder_h1 = Dense(512, activation='relu')
decoder_h2 = Dense(1024, activation='relu')
decoder_mean = Dense(input_dim, activation='sigmoid')

h_decoded = decoder_h1(encoded)
h_decoded = decoder_h2(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)

# Define the autoencoder model
autoencoder = Model(inputs, x_decoded_mean)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))

# Define the regression model using the encoder
encoded_input = Input(shape=(latent_dim,))
regression_h = Dense(256, activation='relu')(encoded_input)
regression_output = Dense(1, activation='linear')(regression_h)

# Regression model
regression_model = Model(encoded_input, regression_output)

# Freeze the encoder layers
for layer in autoencoder.layers[:3]:
    layer.trainable = False

# Extract the encoder part from the autoencoder
encoder = Model(inputs, encoded)

# Get the encoded inputs
encoded_train = encoder.predict(X_train)
encoded_test = encoder.predict(X_test)

# Compile the regression model
regression_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the regression model
regression_model.fit(encoded_train, y_train, epochs=50, batch_size=256, validation_data=(encoded_test, y_test))

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
