# Emotion_Recognition-Neural-Networks::[72.7% Validation Accuracy]
Human facial expressions can be easily classified into 7 basic emotions: happy, sad, surprise, fear, anger, disgust, and neutral. Our facial emotions are expressed through activation of specific sets of facial muscles.
* The aim is to classify the emotion on a person's face into one of **seven categories**, using deep convolution neural networks.
* The algorithm is based on the type of database used inorder to get maximum validation accuracy. Further changes in algorithm may be required according to the database used.

# Dependencies
1. NumPy
2. Keras
3. Pandas
4. Tensorflow (as backend)
5. Jupyter
6. openCv2

# Components
* Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
* Move the downloaded file to the datasets directory inside this repository.
* Trained model Face Detection -> [haarcascade_frontalface_default.xml](https://github.com/piyush2896/Facial-Expression-Recognition-Challenge/blob/master/face_model.h5)
* Trained model JSON -> [model1.h5]
# Algorithm

* The data is fed into training_pixels, training_labels, testing_pixels, testing_labels, respectively.
* The original network starts with an input layer of 48 by 48, matching the size of the input data.
* Then the processing starts with **2 layered convolutions** followed by an intermediate **maxpolling and dropout**
* Further in network it undergoes other **2 layered convolutions** followed by an intermediate **maxpolling and dropout**
* Finally in the network it is **flattened** then **densed** and **dropout** is executed.
* Further the data array undergoes final **dense**, activated by **Softmax** and then **compiled**
* The network is further validated with test data for **16 epoches**.
* Check points for best results are committed in [chkPt1.h5]

* Accuracy Score: 68.68%
# Application in Action
* To see the application in action, by using pretrained JSON model, run the code [App_Interface]
* To train the model using the database and see the results in step by step, run the code blocks in the [Emotion Recognition - Neural Networks-part2.ipynb]
