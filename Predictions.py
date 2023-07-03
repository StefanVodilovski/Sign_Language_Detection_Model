import numpy as np

from Model import model, test_x, test_y
from main import actions

if __name__ == '__main__':

    prediction = model.predict(test_x)

    print(actions[np.argmax(prediction[0])])
    print(actions[np.argmax(test_y[0])])
    model.save('action.h5')