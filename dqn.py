from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D


def createCNN():
    model = Sequential()
    model.add((Conv2D(32, (8, 8), input_shape = (4, 200, 200, 1), strides=4, activation = 'relu'))))
    model.add((Conv2D(64, (4, 4), strides=2, activation = 'relu'))))
    model.add((Conv2D(64, (3, 3), strides=1, activation = 'relu'))))
    model.add(Flatten())
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 8, activation = 'linear'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

    return model
