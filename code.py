from matplotlib import pyplot as plt
import numpy as numpy_alias
import pandas as pandas_alias


class SimpleRNNModel:

    def __init__(self, data, weight1, weight2, bias1, bias2):
        # this includes data preprocessing
        self.rows, self.columns = data.shape
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias1 = bias1
        self.bias2 = bias2
        self.store = {0: 0}

    def ReLU_activation(self, exponent):
        return numpy_alias.maximum(exponent, 0)

    def softmax_activation(self, exponent):
        result = numpy_alias.exp(exponent) / sum(numpy_alias.exp(exponent))
        return result

    def forward(self, X_train, i):
        if i == 0:
            Z1 = self.weight1.dot(X_train) + self.bias1
        else:
            Z1 = numpy_alias.dot(self.weight1, X_train, self.store[i - 1]) + self.bias1
        Affine_1 = self.ReLU_activation(Z1)
        self.store[i] = Affine_1
        Z2 = self.weight2.dot(Affine_1) + self.bias2
        Affine_2 = self.softmax_activation(Z2)

        return Z1, Affine_1, Z2, Affine_2

    def ReLU_derivative(self, exponent):
        return exponent > 0

    def encoding(self, train_Y):
        encoded_Y_train = numpy_alias.zeros((train_Y.size, train_Y.max() + 1))
        encoded_Y_train[numpy_alias.arange(train_Y.size), train_Y] = 1
        encoded_Y_train = encoded_Y_train.T
        return encoded_Y_train

    def backward(self, Z1, Affine_1, Z2, Affine_2, X_train, Y_train):
        encoded_Y = self.encoding(Y_train)
        derivative_Z2 = Affine_2 - encoded_Y
        derivative_weight2 = 1 / self.rows * derivative_Z2.dot(Affine_1.T)
        derivative_bias2 = 1 / self.rows * numpy_alias.sum(derivative_Z2)
        derivative_Z1 = self.weight2.T.dot(derivative_Z2) * self.ReLU_derivative(Z1)
        derivative_weight1 = 1 / self.rows * derivative_Z1.dot(X_train.T)
        derivative_bias1 = 1 / self.rows * numpy_alias.sum(derivative_Z1)
        return derivative_weight1, derivative_bias1, derivative_weight2, derivative_bias2

    def update(self, derivative_weight1, derivative_bias1, derivative_weight2, derivative_bias2, alpha):
        self.weight1 = self.weight1 - alpha * derivative_weight1
        self.bias1 = self.bias1 - alpha * derivative_bias1
        self.weight2 = self.weight2 - alpha * derivative_weight2
        self.bias2 = self.bias2 - alpha * derivative_bias2

    def cal_pred(self, Affine_2):
        return numpy_alias.argmax(Affine_2, 0)

    def calculate_acc(self, predicted_values, train_Y):
        print(predicted_values, train_Y)
        return numpy_alias.sum(predicted_values == train_Y) / train_Y.size

    def train(self, X_train, train_Y, al, no_of_loops):

        for i in range(no_of_loops):
            Z1, Affine_1, Z2, Affine_2 = self.forward(X_train, i)
            derivative_weight1, derivative_bias1, derivative_weight2, derivative_bias2 = self.backward(Z1, Affine_1, Z2, Affine_2, X_train, train_Y)
            self.update(derivative_weight1, derivative_bias1, derivative_weight2, derivative_bias2, al)
            if i % 10 == 0:
                print("No of iteration: ", i)
                derivted_pred = self.cal_pred(Affine_2)
                print(self.calculate_acc(derivted_pred, train_Y))

    def make_predictions(self, X_train):
        Z1 = self.weight1.dot(X_train) + self.bias1
        Affine_1 = self.ReLU_activation(Z1)
        Z2 = self.weight2.dot(Affine_1) + self.bias2
        Affine_2 = self.softmax_activation(Z2)
        predictions = self.cal_pred(Affine_2)
        return predictions

    def test_prediction(self, index, X_test, Y_test):
        current_image = X_test[:, index, None]
        prediction = self.make_predictions(X_test[:, index, None])
        label = Y_test[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()


data_frame = pandas_alias.read_csv('https://personal.utdallas.edu/~yxn210005/train.csv')

rows, columns = data_frame.shape

weight1 = numpy_alias.random.rand(10, 784) - 0.5
bias1 = numpy_alias.random.rand(10, 1) - 0.5
weight2 = numpy_alias.random.rand(10, 10) - 0.5
bias2 = numpy_alias.random.rand(10, 1) - 0.5

data_frame = numpy_alias.array(data_frame)
numpy_alias.random.shuffle(data_frame)
train_data = data_frame[1000:rows].T
train_Y = train_data[0]
temp_X_train = train_data[1:columns]
train_X = temp_X_train / 255.

data_test = data_frame[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:columns]
X_test = X_test / 255.

simpleRNNModel = SimpleRNNModel(data_frame, weight1, weight2, bias1, bias2)
simpleRNNModel.train(train_X, train_Y, 0.20, 800)
simpleRNNModel.test_prediction(0, train_X, train_Y)
simpleRNNModel.test_prediction(1, train_X, train_Y)
simpleRNNModel.test_prediction(2, train_X, train_Y)
simpleRNNModel.test_prediction(3, train_X, train_Y)

simpleRNNModel.calculate_acc(simpleRNNModel.make_predictions(X_test), Y_test)
