# Sales Prediction Using Deep Feedforward Neural Network
## Project Overview
This project implements a deep feedforward neural network to predict Sales using structured data that includes both categorical and numerical features. The aim is to learn complex relationships between business-relevant features and sales outcomes using deep learning techniques.
## Model Architecture
➤ Architecture Type <br>
Feedforward Neural Network (Multilayer Perceptron) designed for regression.

➤ Input Features <br>
Shipping Mode

Segment

City

State

Region

Category

Sub-Category

Shipping Time

All categorical features were label encoded, and numerical features were standard scaled before training. <br>

➤ Network Layers <br>

Input Layer	8 Features <br>
Hidden Layer 1	64 Neurons, ReLU Activation <br>
Hidden Layer 2	32 Neurons, ReLU Activation <br>
Output Layer	1 Neuron, Linear Activation (for Sales) <br>


## Training Configuration
Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Evaluation Metrics:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Batch Size: 32

Train-Test Split: 80% Train / 20% Test

Regularization: Dropout / Early Stopping (experimentally used)

## Training Process
Trained using batch gradient descent and backpropagation

Tracked training progress using loss curves

Adjusted hyperparameters to improve convergence

Training continued until validation loss plateaued

## Evaluation Metrics
After training, the model was evaluated on the test set using the following:

MAE: Measures average prediction error

RMSE: Penalizes larger errors

R² Score: Indicates the proportion of variance in the target variable that is predictable from the features

## Results and Visualization
Plotted loss curves to monitor learning behavior

Evaluated model performance across different error metrics to assess overall accuracy and generalization

The model showed promising results in capturing nonlinear patterns for sales prediction
