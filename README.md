# ABHIROCKZ
EMOTION RECOGNITION 
Emotion_Recognition--Neural-Networks::[68.68% Validation Accuracy]
Human facial expressions can be easily classified into 7 basic emotions: happy, sad, surprise, fear, anger, disgust, and neutral. Our facial emotions are expressed through activation of specific sets of facial muscles.

The aim is to classify the emotion on a person's face into one of seven categories, using deep convolution neural networks.
The algorithm is based on the type of database used inorder to get maximum validation accuracy. Further changes in algorithm may be required according to the database used.
Dependencies
NumPy
Keras
Pandas
Tensorflow (as backend)
Jupyter
openCv2
Components
Download the fer2013.tar.gz file from here
Move the downloaded file to the datasets directory inside this repository.
Trained model Face Detection -> haarcascade_frontalface_default.xml
Trained model JSON -> model1.h5
Algorithm
Database distribution is something like:

The data is fed into training_pixels, training_labels, testing_pixels, testing_labels, respectively.
The original network starts with an input layer of 48 by 48, matching the size of the input data.
Then the processing starts with 2 layered convolutions followed by an intermediate maxpolling and dropout
Further in network it undergoes other 2 layered convolutions followed by an intermediate maxpolling and dropout
Finally in the network it is flattened then densed and dropout is executed.
Further the data array undergoes final dense, activated by Softmax and then compiled
The network is further validated with test data for 16 epoches.
Check points for best results are committed in chkPt1.h5
Summary of convolution neural networks applied over the database:

Result after 16 epoches :

Accuracy Score: 68.68%
Application in Action
To see the application in action, by using pretrained JSON model, run the code App_Interface
To train the model using the database and see the results in step by step, run the code blocks in the Emotion Recognition - Neural Networks-part2.ipynb
