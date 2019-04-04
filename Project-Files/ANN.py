import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
Y_train = np.load('Y_train.npy')
Y_test = np.load('Y_test.npy')
print(X_train.shape," ",X_train.dtype)
print(X_test.shape," ",X_test.dtype)
print(Y_train.shape," ",Y_train.dtype)
print(Y_test.shape," ",Y_test.dtype)
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]
def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(num_pixels/2), kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(num_pixels/4), kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(num_pixels/8), kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(num_pixels/16), kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(num_pixels/32), kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(num_pixels/64), kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = baseline_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("ANN Baseline Error: %.2f%%" % (100-scores[1]*100))
