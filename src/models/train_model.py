import os

import numpy as np
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense


def get_actions():
    files = os.listdir('../../Data/processed/MP_DATA/')
    return np.array(files)


def train_model(train_x, train_y):
    actions = get_actions()

    log_dir = os.path.join('logs')
    callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(train_x, train_y, epochs=90, callbacks=[callback])
    model.summary()
    n = len(actions)
    model.save(f'C:\\Users\\vodil\\PycharmProjects\\Sign Language Detection\\Models\\action({n}).h5')
    print(f'Model saved as action{n}.h5')
