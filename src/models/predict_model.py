import keras.models
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


def predict_model(model_file,test_x,test_y):
    model = keras.models.load_model(model_file)
    yhat = model.predict(test_x)
    ytrue = np.argmax(test_y, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    multilabel_confusion_matrix(ytrue, yhat)
    print(accuracy_score(ytrue, yhat))