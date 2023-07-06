import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from Model import model,model_2, test_x, test_y, train_x, train_y
from main import actions

# if __name__ == '__main__':
#     #
#     #
#     # del model
#     # model.load_weights('action.h5')
#
#     # print("train x :")
#     # print(train_x.shape)
#     # print("test_x:")
#     # print(test_x.shape)
#     # print("train_y:")
#     # print(train_y.shape)
#     # print("test_y:")
#     # print(test_y.shape)