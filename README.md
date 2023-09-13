# A Sign Language Detection model based on a LSTM (Long short-term memmory) neural network 
#### This is a model for detectiong actions trained on ASL sign language data.
## 1) How to start the project
   --- You need to clone this repo:
````
   git clone [https://github.com/StefanVodilovski/sign-language-detection.git](https://github.com/StefanVodilovski/Sign_Language_Detection_Model.git)
````
   -- Write the following command to install all dependencies:
````

   pip install requirements

````
## 2) Model predictions
   -For now the model can detect only 4 actions (hello, i love you, thanks and please )
   -If You want to add new actions go to add_action.py and generate data for training
   -You can try out the real time predictions when running real_time_predict.py

## 3) Data
   -if you want to add data create the following path Data/proccesed/MP_DATA
   -the model in Models is trained for 5 predictions, actions for hello, i love you, thanks, please and no action(when you stand/ sit still)

## 4) the src folder
   - In the src folder there are 4 major files:
   - The main.py file:
   - This is the file where we initialize the starting 3 actions and create data for them so we can train the model with
   - The model.py file:
   - Training the model on the created data
   - Real_Time_Predict.py
   - Get the coresponding model stored in the Models file and launch the camera for detection (you can modify the threshold for better accuracy when predicting) 
