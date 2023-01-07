# Image-Classification-with-Artificial-Neural-Networks
Abstract—In this document, we discuss about the classification of images using a Recurrent Neural Network (RNN) model on MNIST data of handwritten digits. To evaluate the performance of the model, Mean Average Precision (mAP) is used.

Index Terms — Image classification, RNN, Accuracy, Machine Learning


I.	INTRODUCTION
Image classification is a supervised learning problem, it defines a set of target classes and train the model to recognize them using labeled examples. The application of the image classification using machine learning includes machine vision, brake light detection, traffic control systems, object identification in satellite images, medical imaging and much more.
Recurrent Neural Network (RNN) is one if the earliest computer algorithm that has the memory and it remembers the input information. While when it comes to humans, we don’t rely much on the brain in order to interpret any statement as we can understand the context of any word based on the previous knowledge and retention power. The above-mentioned attribute of RNN is what that makes it as a powerful model for many real word applications. RNNs are a family of networks that are suitable for learning representations of sequential data like text in natural language processing (NLP) or a stream of sensor data in instrumentation.
Recurrent Neural Network (RNN) is one if the earliest computer algorithm that has the memory and it remembers the input information. While when it comes to humans, we don’t rely much on the brain in order to interpret any statement as we can understand the context of any word based on the previous knowledge and retention power. The above-mentioned attribute of RNN is what that makes it as a powerful model for many real word applications.
In this report, we will be implementing a recurrent neural network-based model for the classification of images. Using this model, we can identify the handwritten numbers accurately from the MNIST dataset.

II.	BACKGROUND WORK
Image classification is a very important input for a countless number of applications that includes facial recognition, driverless cars, medical disease identification. When coming to the image classification a lot research has been carried over the years and many models has been developed. In general, the accuracy for the statistical models is always high. Neural networks are precisely built to solve the complex issues that traditional methods or statistical methods couldn’t solve. Off late, linear models are outperformed by non-linear models.
RNN is popular choice for the sequence processing, object detection and is quite suitable for performing image classification.


Figure. 1. Basic architecture of Recurrent Neural Networks, Source- [3]

III.	THEORETICAL AND CONCEPTUAL STUDY
A.	Classification Of Images
When it comes to image identification, humans can recognize huge number of objects as we train our brain over the years to interpret the data from what we see.
With the advanced technology holding a significant part in     the modern world, machine learning offers enormous potential for the problem solving and has many potential applications. Object detection is one among the many applications of the image classification, object detection is a computer vision technique that works to locate and identify the objects in an image.
Object detection using machine learning are based on various features on the image such as to locate the group of pixels that may belong to an object, color histogram or edges. These features are fed into regression model that predicts the location of the objects in the image along with its label.
 

Fig.2. Object detection with image classification

B.	Recurrent Neural Network (RNN)
Recurrent neural networks concede the linear property of the data, RNN use the patterns to predict most likely outcomes. They are typical artificial neural networks (ANN) that are abundantly utilized in natural language processing and speech recognition.
  RNNs are derived from the first and simple type of artificial                      neural networks called as Feed-Forward Neural Network(FNN).    In FNNs the information flows only in one direction i.e., forward. The data travels in a straight line and never passes through a same node twice through the network as shown in “Fig. 3”.

 

Fig. 3. FNN vs RNN

Feed Forward Neural Network is a poor predictor of the future and doesn’t have the ability to recollect the information they receive. Recurrent neural network depends on the previous stages as input to the future state and to recollect the data from the past. Due to this ability RNNs are aptly used for processing sequences of data. RNN can accept variable-length sequences as inputs and outputs, it learns by updating the weights over a period in order to train the model.

A simple RNN provides the same weights and biases to all the layers for converting the independent activation into the dependent activations, it reduces the complication of the increasing parameters and memorizes the output of each layer by giving it as an input to next hidden layer. The following is the equation of the simple RNN,

 

In this equation b is the bias, while W is the weight for the previous output and called as recurrent kernel. While U is the weights for the current input called as kernel. Subscript t is used to indicate the position in the sequence. The following figure shows the diagrams of both simple RNN and RNN when used for the classification task. The difference between the simple RNN and RNN is the absence of the output values ot = Vht + c before the function is computed.

 

Fig. 4. Simple RNN vs RNN

IV.	MODEL IMPLEMENTATION
A. Data Set
The MNIST database consists of handwritten digits with test set consisting of 10,000 examples and training set consisting of 60,000 samples respectively. The digits in the dataset are centered in a fixed size image and they are size normalized.
NIST’s Special Database 3 and Special Database 1 which has binary images of handwritten digits are used for construction of MNIST database. Initially SD-3 is considered as training set whereas SD-1 is considered as test set by NIST. However, when comparing SD-3 with SD-1, we can say that SD-3 is easy to recognize and is much cleaner than SD-1. The fact that contributes to it is that SD-1 was collected from the high school students and SD-3 was collected from the census bureau employees. The results must be independent of choice of training set and the test set among the set of samples in order to obtain meaningful conclusions. Therefore, it needs to necessary situation of building a new database by combining all NIST’s datasets.
The MNIST training set consists of 30,000 samples from SD-1 and 30,000 samples from SD-3. Whereas the test set have 5,000 samples from SD-1 and SD-3 each. Approximately 250 writers’ samples were considered for the training set. 58,527-digit images written by 500 different writers was considered for SD-1. In contrast to SD-3, where blocks of data from each writer appeared in sequence, SD-1 has scrambled data and writer identities for SD-1 is there and this information is used to unscramble the writers. Then SD-1 is split in to two: characters written by the first 250 writers went into new training set. The remaining 250 writers were placed in test set. Thus, it had two sets with nearly 30,000 examples each. The new training set was completed with enough examples from SD-3.

B.	Implementation
For image classification, we have implemented RNN with forward and back propagation algorithm on our own using basic packages of python programming language such as pandas and NumPy. Figure 5 shows block representation of the model where a 28 × 28 image is sent to RNN model which predicts the class of digit specified in the image i.e., 0-9. The entire data has been split into training data and testing data with a ratio of 80:20 respectively. RNN is mostly used for sequential data, but the MNIST data is not sequential, and it is also not feasible to interpret every image as a sequence of rows or columns of pixel. So, the images are processed as a sequence of 28 element input vectors with time steps equal to 28. From the simple RNN equation, we defined two vectors W1 and W2 where W1 is called recurrent kernel weights for the previous output and W2 is kernel weights for the current input. The RNN model we designed is a many to one model which produces a single value output i.e., the value of the number specified in the image. The learning rate we choose is 0.01 with 500 epochs. The activation function we used for the hidden layer of our RNN model is relu activation.

 
Fig 5: RNN model for MNIST digit classification

We defined functions for both forward and back propagation with the previous value being stored for every iteration of forward pass such that they can be used for recomputing or updating the weights for the forthcoming iterations. In forward pass, for the first iteration we multiply the weights with input whereas from the next iterations we multiplied the weights and input with the previous iteration values. Then we passed it to a relu activation. Now, we use the backward propagation to update the weights until the algorithm gets converged or it reaches the epochs specified at instantiation. Finally, we trained our data using our RNN model and made predictions.
C.	Evaluation
For a classification model in machine learning, the most significant and accurate performance evaluation metric is accuracy. The evaluation metric we used for our use case is Mean average precision. The mathematical formula for calculating the mean average precision of a machine learning model is shown in Figure 6. 
 
Fig. 6. Accuracy formula
The value of accuracy lies between 0 and 1 where 1 is the ideal case where there are zero misclassified samples and 0 is the worst-case scenario where number of misclassified samples is equal to the total number of samples i.e., no sample being predicted correctly. The model we designed achieved an accuracy of 0.8.


V.	RESULTS AND ANALYSIS
The model evaluation for our RNN prediction model based on Mean average precision which has been calculated based on the formula in Figure. Based on the number of misclassified samples, the accuracy varies between 0 to 1. We have measured the mean average precision for each 10 iterations of the total 500 epochs. The accuracy seemed to be increasing significantly for every measurement which shows that the model is making good progress in prediction.

Epochs	Accuracy
100	0.6471
200	0.7703
300	0.8146
400	0.8363
500	0.8610
Fig. 7. Model Results

The table in Figure 7 shows the accuracy of the model over epochs. From this, as the accuracy increases on increasing epochs, we can say that the model is feasible for the data set. After 500 epochs, the accuracy of the model is 0.861 which is a very reasonable and acceptable value for accuracy. Figure 8 shows the sample image for digit classification with the class or label being predicted by the model.
 
 
Fig. 8. Model prediction

VI.	CONCLUSION AND FUTURE WORK
In this paper, we predicted the class digit that is specified in a sample image using an RNN model. We designed the RNN model with a hidden layer for 500 epochs and learning rate of 0.01. The 28×28-pixel image has been sent to the model as a sequence of 28 element input vectors with time steps equal to 28. We have recorded the model prediction accuracy for every 10 iterations where we have witnessed an increasing accuracy over epochs. The model prediction accuracy was acceptable with a value of 0.861. 
In our RNN model prediction, we have confined to 2D image which is a grey scale image. But as a future enhancement a model for prediction of 3D images i.e., RGB image, can also be built, for which the concept of RNN may not be a correct fit. For such type of images, CNN (Convolutional Neural Networks) model is used which includes much more complex concepts such as convolution, images filtering and pooling techniques.
In CNN, the image is passed through three layers namely Convolutional layer, Pooling layer and fully connected Neural net. The Convolutional layer reduces the dimensionality of the image using various filters. The pooling layer is purely for computational point of view. It allows either the average or maximum value from the result, depending on the application. And finally, the data from pooled layer is sent to a neural network which further classified the images based on the applications.

REFERENCES:
[1]	https://subscription.packtpub.com/book/programming/9781838821654/1/ch01lvl1sec06/5-recurrent-neural-network-rnn
[2]	https://thegradient.pub/a-visual-history-of-interpretation-for-image-recognition/
[3]	https://www.tensorflow.org/guide/keras/rnn
[4]	https://towardsdatascience.com/a-practical-guide-to-rnn-and-lstm-in-keras-980f176271bc
[5]	https://thecleverprogrammer.com/2021/07/01/calculation-of-accuracy-using-python/
[6]	https://www.analyticsvidhya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/

