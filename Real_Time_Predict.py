import cv2
import keras.models
import numpy as np


from main import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints, actions

model = keras.models.load_model("action.h5")
if __name__=='__main__':
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.4

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)

            #prediction
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence,axis=0))[0]
                print(res)
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
            #visuals
                if(np.unique(predictions[-10:]))[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            cv2.rectangle(image,(0,0), (640,40), (245,117,16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


            #Show to screen
            cv2.imshow('OpenCV Feed', image)  # show

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()