from src.Data.processed_to_sets import processed_to_sets
from src.Data.stream_to_keypoints import stream_to_keypoints
from src.models.train_model import train_model

if __name__ == '__main__':
    print("Add action name")
    action = input()
    stream_to_keypoints(action,90,30)
    train_x,test_x,train_y,test_y = processed_to_sets(30)
    train_model(train_x,train_y)





