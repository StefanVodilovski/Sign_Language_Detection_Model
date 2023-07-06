import os

import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import TensorBoard

from main import DATA_PATH, sequence_length, no_sequences, actions


# preprocess data, train set, test set
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        #get all frames in a window
        for frame_number in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH,action, str(sequence), "{}.npy".format(frame_number)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


X = np.array(sequences)
print(X.shape)
Y = to_categorical(labels).astype(int)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.10)

log_dir= os.path.join('logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128,return_sequences=True, activation='relu'))
model.add(LSTM(64,return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])



if __name__ == '__main__':

    model.fit(train_x, train_y, epochs=90, callbacks=[tb_callback])
    model.summary()

    yhat = model.predict(test_x)
    ytrue = np.argmax(test_y, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    multilabel_confusion_matrix(ytrue, yhat)
    print(accuracy_score(ytrue, yhat))

    model.save('action.h5')




