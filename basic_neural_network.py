
# Importing the numpy library for numerical operations
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Defining a class NeuralNetwork
class NeuralNetwork:
    def __init__(self,X):
        a=X.shape[1]
        b=X.shape[1]
        # Setting a fixed random seed to ensure reproducibility of results
        np.random.seed(10)
        # Initializing weights from the input layer to the hidden layer (3 input neurons to 4 hidden neurons)
        self.wij = np.random.rand(a, b)
        # Initializing weights from the hidden layer to the output layer (4 hidden neurons to 1 output neuron)
        self.wjk = np.random.rand(b, 1)

    # Sigmoid activation function: maps any value to a value between 0 and 1
    def sigmoid(self, x, w):
        z = np.dot(x, w)  # Calculating the weighted sum of input and weights
        return 1 / (1 + np.exp(-z))  # Applying the sigmoid function

    # Derivative of the sigmoid function, used in backpropagation
    def sigmoid_derivative(self, x, w):
        return self.sigmoid(x, w) * (1 - self.sigmoid(x, w))

    # Function to perform gradient descent optimization
    def gradient_descent(self, x, y, iterations):
        loss_history = []  # Initializing an empty list to store the loss at each iteration
        accuracy_history = []
        for i in range(iterations):
            # Forward propagation
            Xi = x  # Input layer activations
            Xj = self.sigmoid(Xi, self.wij)  # Hidden layer activations
            yhat = self.sigmoid(Xj, self.wjk)  # Output layer predictions

            # Backpropagation
            # Calculating gradient for the weights between hidden and output layer
            g_wjk = np.dot(Xj.T, (y - yhat) * self.sigmoid_derivative(Xj, self.wjk))
            # Calculating gradient for the weights between input and hidden layer
            g_wij = np.dot(Xi.T, np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.wjk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij))

            # Updating weights
            self.wij += g_wij
            self.wjk += g_wjk

            # Calculate and store loss
            loss = np.mean(np.square(y - yhat))
            loss_history.append(loss)

            # Calculate and store accuracy
            predictions = np.round(yhat)  # Rounding off predictions for binary classification
            accuracy = np.mean(predictions == y)  # Comparing with actual labels
            accuracy_history.append(accuracy)

        return loss_history, accuracy_history


def main():
    st.title("Basic Neural Network")
    st.write("This is a basic neural network that can be trained on a dataset with x input features and 1 output label. The neural network has 1 hidden layer and uses sigmoid activation function. The neural network is trained using gradient descent optimization for 10000 iterations.")
    # Sidebar for file upload
    upload_file = st.sidebar.file_uploader("Choose a file", type=['csv'])
    st.sidebar.write("The uploaded file should have x input features and 1 output label. The input features should be separated by a semicolon (;) and the output label should be in the last column.")
    if upload_file is not None:
        # Reading the uploaded file
        data = pd.read_csv(upload_file, delimiter=';')
        st.write("Uploaded DataFrame:")
        st.dataframe(data)

        # Data preparation
        X = data.iloc[:,1:-1].values 
        y = data.iloc[:,-1].values.reshape(-1,1)
        # Creating an instance of the NeuralNetwork class
        neural_network = NeuralNetwork(X)

        # Button to start training
        if st.button("Train Neural Network"):
            # Displaying initial weights
            st.write('Random starting input to hidden weights: ')
            st.write(neural_network.wij)
            st.write('Random starting hidden to output weights: ')
            st.write(neural_network.wjk)

            # Running the gradient descent optimization for 10000 iterations
            loss_history, accuracy_history = neural_network.gradient_descent(X, y, 10000)

            # Displaying the final predictions after training
            st.write('The final prediction from neural network are: ')
            yhat = np.round(neural_network.sigmoid(neural_network.sigmoid(X, neural_network.wij), neural_network.wjk), 0)
            st.dataframe(yhat, hide_index=False)

            # Plotting the loss history
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history, label='Loss per iteration')
            plt.plot(accuracy_history, label='Accuracy per iteration')
            plt.xlabel('Iterations')
            plt.title('Gradient Descent')
            plt.legend()
            st.pyplot(plt)


# Main execution block
if __name__ == '__main__':
    main()