import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
#import mnist data
from keras.datasets import mnist
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

dataSize = x_train.shape[1:3]
nClasses = np.unique(y_train).shape[0]

# Reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot encode targets
y_train = keras.utils.to_categorical(y_train, nClasses)
y_test = keras.utils.to_categorical(y_test, nClasses)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(dataSize[0], dataSize[1], 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

kerasClassifier = KerasClassifier(build_fn=model, epochs=10, batch_size=200, verbose=1)
cross_val_score(kerasClassifier, x_train, y_train, cv=3)

