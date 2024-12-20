import os
import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Function for converting units from imperial to SI
def convert_to_si(PLA, N1, T30, Wf):
    """
    Convert units from imperial to SI.

    Parameters:
    - PLA: Throttle angle in degrees (no conversion, but kept for consistency)
    - N1: N1 speed in rpm (no conversion, but kept for consistency)
    - T30: Temperature in °R (Rankine)
    - Wf: Fuel flow in pps (pounds per second)

    Returns:
    - PLA_SI: Throttle angle in degrees (no conversion applied)
    - N1_SI: N1 speed in rpm (no conversion applied)
    - T30_SI: Temperature in K (Kelvin)
    - Wf_SI: Fuel flow in kg/s
    """
    # Convert throttle angle from degrees
    PLA_SI = PLA

    # N1 is in rpm, we keep it as is (no conversion needed)
    N1_SI = N1

    # Convert temperature from °R (Rankine) to K (Kelvin)
    T30_SI = T30 * (5.0 / 9.0)

    # Convert fuel flow from pps (pounds per second) to kg/s
    Wf_SI = Wf * 0.453592

    return PLA_SI, N1_SI, T30_SI, Wf_SI


# Loading CMAPSS data from the HDF5 file
file_path = 'N-CMAPSS_DS02-006.h5'

# Set the number of samples to use (e.g., 10000)
num_samples = 500000

with h5py.File(file_path, 'r') as hdf:
    # Extract training and test datasets
    Xs_dev = np.array(hdf.get('X_s_dev'))[:num_samples]  # Sensor readings for training
    W_dev = np.array(hdf.get('W_dev'))[:num_samples]     # Operational conditions for training
    Xs_test = np.array(hdf.get('X_s_test'))[:num_samples]  # Sensor readings for testing
    W_test = np.array(hdf.get('W_test'))[:num_samples]     # Operational conditions for testing

# --- Preparation of training data (dev) ---
T30_dev = Xs_dev[:, 1]             # T30 temperature in °R (Rankine)
Wf_col_dev = Xs_dev[:, 13]         # Wf (Fuel flow) in pps (pounds per second)
throttle_angle_dev = W_dev[:, 2]   # Throttle angle in degrees
N1_dev = Xs_dev[:, 11]             # Nf (used as N1 speed) in rpm

# Unit conversion for the training set
PLA_SI_dev, N1_SI_dev, T30_SI_dev, Wf_SI_dev = convert_to_si(throttle_angle_dev, N1_dev, T30_dev, Wf_col_dev)

# Preparation of features and target for the training set
features_dev = np.column_stack((PLA_SI_dev, N1_SI_dev, T30_SI_dev))
target_dev = Wf_SI_dev.reshape(-1, 1)

# --- Preparation of test data ---
T30_test = Xs_test[:, 1]             # T30 temperature in °R (Rankine)
Wf_col_test = Xs_test[:, 13]         # Wf (Fuel flow) in pps (pounds per second)
throttle_angle_test = W_test[:, 2]   # Throttle angle in degrees
N1_test = Xs_test[:, 11]             # Nf (used as N1 speed) in rpm

# Unit conversion for the test set
PLA_SI_test, N1_SI_test, T30_SI_test, Wf_SI_test = convert_to_si(throttle_angle_test, N1_test, T30_test, Wf_col_test)

# Preparation of features and target for the test set
features_test = np.column_stack((PLA_SI_test, N1_SI_test, T30_SI_test))
target_test = Wf_SI_test.reshape(-1, 1)

# Normalization of training and test data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_normalized_dev = scaler_X.fit_transform(features_dev)
y_normalized_dev = scaler_y.fit_transform(target_dev)

X_normalized_test = scaler_X.transform(features_test)
y_normalized_test = scaler_y.transform(target_test)

# --- Linear Regression Model ---
model_lr = LinearRegression()
model_lr.fit(X_normalized_dev, y_normalized_dev)  # Training the model

# Prediction of fuel flow using linear regression
predicted_lr_normalized = model_lr.predict(X_normalized_test)
predicted_lr_SI = scaler_y.inverse_transform(predicted_lr_normalized)  # Inverse transform to SI units
y_test_SI = scaler_y.inverse_transform(y_normalized_test)

# Mean Squared Error for Linear Regression
mse_lr = mean_squared_error(y_test_SI, predicted_lr_SI)
print(f'Mean Squared Error (Linear Regression): {mse_lr:.6f}')


# --- Neural Network Model ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Conversion of data to tensors for PyTorch
X_train_tensor = torch.tensor(X_normalized_dev, dtype=torch.float32)
y_train_tensor = torch.tensor(y_normalized_dev, dtype=torch.float32)
X_test_tensor = torch.tensor(X_normalized_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_normalized_test, dtype=torch.float32)

# Definition of the neural network, loss function, and optimizer
input_dim = X_train_tensor.shape[1]
output_dim = 1
model_nn = SimpleNN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.01)

# Training the neural network
num_epochs = 1000
train_loss = []
for epoch in range(num_epochs):
    model_nn.train()
    optimizer.zero_grad()
    outputs = model_nn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# Evaluating the neural network
model_nn.eval()
with torch.no_grad():
    predicted_nn_normalized = model_nn(X_test_tensor).numpy()

# Inverse transformation of predictions and targets to SI units
predicted_nn_SI = scaler_y.inverse_transform(predicted_nn_normalized)

# Mean Squared Error for the Neural Network
mse_nn = mean_squared_error(y_test_SI, predicted_nn_SI)
print(f'Mean Squared Error (Neural Network): {mse_nn:.6f}')

# --- Visualization of Results ---
num_samples_to_plot = 500000

# Plotting actual and predicted values for both models
fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Neural Network
axs[0].plot(y_test_SI[:num_samples_to_plot], label='Actual Wf (Test Set)', color='blue', linewidth=2.5)
axs[0].plot(predicted_nn_SI[:num_samples_to_plot], label='Predicted Wf (Neural Network)', color='red', linestyle='--', linewidth=2.5, alpha=0.8)
axs[0].set_title('Actual vs Predicted Fuel Flow (Wf) - Neural Network', fontsize=14)
axs[0].set_ylabel('Fuel Flow (kg/s)', fontsize=12)
axs[0].legend(fontsize=12)
axs[0].grid(True, linestyle='--', alpha=0.6)

# Linear Regression
axs[1].plot(y_test_SI[:num_samples_to_plot], label='Actual Wf (Test Set)', color='blue', linewidth=2.5)
axs[1].plot(predicted_lr_SI[:num_samples_to_plot], label='Predicted Wf (Linear Regression)', color='orange', linestyle='--', linewidth=2.5, alpha=0.8)
axs[1].set_title('Actual vs Predicted Fuel Flow (Wf) - Linear Regression', fontsize=14)
axs[1].set_xlabel('Sample Index', fontsize=12)
axs[1].set_ylabel('Fuel Flow (kg/s)', fontsize=12)
axs[1].legend(fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Plotting residuals for both models
residuals_nn_SI = predicted_nn_SI - y_test_SI
residuals_lr_SI = predicted_lr_SI - y_test_SI
plt.figure(figsize=(14, 6))
plt.plot(residuals_nn_SI[:num_samples_to_plot], label='Residuals (Neural Network)', color='red', linestyle='--', linewidth=2, alpha=0.8)
plt.plot(residuals_lr_SI[:num_samples_to_plot], label='Residuals (Linear Regression)', color='orange', linestyle='--', linewidth=2, alpha=0.8)
plt.title('Residuals (Predicted - Actual)', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Residuals (kg/s)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Distribution of residuals for both models
plt.figure(figsize=(10, 6))
plt.hist(residuals_nn_SI, bins=30, alpha=0.6, label='Neural Network Residuals', color='red', edgecolor='black')
plt.hist(residuals_lr_SI, bins=30, alpha=0.6, label='Linear Regression Residuals', color='orange', edgecolor='black')
plt.title('Distribution of Residuals in SI Units', fontsize=14)
plt.xlabel('Residual Value (kg/s)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

