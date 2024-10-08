from tensorflow import keras
import os

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

# download Cifar-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# print the size of the dataset
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[0])
# Convert the category label into onehot encoding
num_classes = 10
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes)
y_train[0]

# Create a image tag list
category_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog',
                 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
# Show the first 9 images and their labels
plt.figure()
for i in range(9):
    # create a figure with 9 subplots
    plt.subplot(3, 3, i + 1)
    # show an image
    plt.imshow(x_train[i])
    # show the label
    plt.ylabel(category_dict[y_train[i][0]])
plt.show()

# Pixel normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# Model Creation

def CNN_classification_model(input_size=x_train.shape[1:]):
    model = Sequential()
    # the first block with 2 convolutional layers and 1 maxpooling layer
    '''Conv1 with 32 3*3 kernels 
        padding="same": it applies zero padding to the input image so that the input image gets fully covered by the filter and specified stride.
        It is called SAME because, for stride 1 , the output will be the same as the input.
        output: 32*32*32'''
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_size))
    # relu activation function
    model.add(Activation('relu'))
    # Conv2
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    # maxpooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    # the second block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # maxpooling.the default strides =1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Before sending a feature map into a fully connected network, it should be flattened into a column vector.
    model.add(Flatten())
    # fully connected layer
    model.add(Dense(128))
    model.add(Activation('relu'))
    # dropout layer.every neuronis set to 0 with a probability of 0.25
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    # map the score of each class into probability
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


model = CNN_classification_model()
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

from tensorflow.keras.callbacks import ModelCheckpoint

model_name = "final_cifar10.h5"
model_checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)

# load pretrained models
trained_weights_path = 'cifar10_weights.h5'
if os.path.exists(trained_weights_path):
    model.load_weights(trained_weights_path, by_name=True)
# train
model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=[model_checkpoint], verbose=1)
