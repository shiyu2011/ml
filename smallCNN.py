import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical

#load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

#normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

#one-hot encode the labels
train_labels = to_categorical(train_labels)

#build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))#n x 28 x 28 x 1 -> n x 26 x 26 x 32
model.add(MaxPooling2D((2, 2))) #n x 26 x 26 x 32 -> n x 13 x 13 x 32
#flatten from 2D to 1D
model.add(Flatten()) #n x 13 x 13 x 32 -> n x 5408
#add fully connected layer
model.add(Dense(64, activation= 'relu')) #n x 5408 -> n x 64
#add output layer   
model.add(Dense(10, activation= 'softmax')) #n x 64 -> n x 10

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#add callback of tensorboard, and save models every epoch, display the loss and accuracy
from keras.callbacks import TensorBoard
import os
logdir = os.path.join("logs", "smallCNN")
tensorboard = TensorBoard(log_dir=logdir)
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1, callbacks=[tensorboard])
#train the model
test_loss, test_acc = model.evaluate(test_images, to_categorical(test_labels))
print('Test accuracy:', test_acc)
#sensitivity analysis
import shap
#select a set of background examples to take an expectation over
background = train_images[np.random.choice(train_images.shape[0], 100, replace=False)]
#explain predictions of the model on four images
e = shap.DeepExplainer(model, background)
#...or pass tensors directly
#e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(test_images[1:5])
