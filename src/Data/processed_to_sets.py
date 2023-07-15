import os

import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("../../Data/processed/MP_DATA/")


def get_actions():
    files = os.listdir('../../Data/processed/MP_DATA/')
    return np.array(files)


def processed_to_sets(sequence_length):
    actions = get_actions()

    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            # get all frames in a window
            for frame_number in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_number)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    print(X.shape)
    Y = to_categorical(labels).astype(int)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.10)
    return train_x, test_x, train_y, test_y
