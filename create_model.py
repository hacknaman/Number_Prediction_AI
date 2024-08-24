import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from utils import *

# load dataset
X, y = load_data()
m, n = X.shape

np.set_printoptions(precision=2, suppress=True, linewidth=150)

print ('X type: ', X.dtype )
print ('The first element of X is: \n', X[0].reshape(20,20) )
print ('The first element of y is: ', y[0,0])
print ('The first element of X is: \n', X[1015].reshape(20,20) ) # this is 2
print ('The first element of y is: ', y[1015,0])
print ('The Last element of X is: \n', X[-1].reshape(20,20) )
print ('The last element of y is: ', y[-1,0])

print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))


fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20))
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Display the label above the image
    ax.set_title(y[random_index,0])
    ax.set_axis_off()

fig.suptitle("Label and Image", fontsize=14)
plt.show()

# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(45, activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(10, activation = 'linear')  
    ], name = "my_model" 
)
model.summary()

[layer1, layer2, layer3] = model.layers

#### Examine Weights shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=50
)

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20))
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1,400))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
    ax.set_axis_off()

fig.suptitle("Label, Prediction", fontsize=14)
plt.show()

print( f"saving model" )
model.save('my_model.keras')

print( f"{display_errors(model,X,y)} errors out of {len(X)} images")