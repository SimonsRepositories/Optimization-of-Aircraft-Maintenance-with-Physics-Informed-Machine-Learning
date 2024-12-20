import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import h5py

# Step 1: Load data from the exported residuals from the GNN script
with h5py.File("data_exports/metadata_residuals.h5", "r") as h5f:
    # train
    residuals_train_data = np.array(h5f['residuals_train'])
    ruls_train_data = np.array(h5f['ruls_train'])
    cycles_train_data = np.array(h5f['cycles_train'])
    units_train_data = np.array(h5f['units_train'])
    # test
    residuals_test_data = np.array(h5f['residuals_test'])
    ruls_test_data = np.array(h5f['ruls_test'])
    cycles_test_data = np.array(h5f['cycles_test'])
    units_test_data = np.array(h5f['units_test'])

# Helper function to truncate data by RUL
def truncate_data_by_rul(residuals, ruls, cycles, units):
    truncated_residuals = []
    truncated_ruls = []
    truncated_cycles = []
    truncated_units = []

    unique_units = np.unique(units)
    for unit in unique_units:
        unit_indices = units == unit
        unit_ruls = ruls[unit_indices]
        unit_cycles = cycles[unit_indices]
        unit_residuals = residuals[unit_indices]

        # Find the point where RUL = 1 and truncate
        valid_length = np.argmax(unit_ruls == 1) + 1 if np.any(unit_ruls == 1) else len(unit_ruls)
        truncated_residuals.append(unit_residuals[:valid_length])
        truncated_ruls.append(unit_ruls[:valid_length])
        truncated_cycles.append(unit_cycles[:valid_length])
        truncated_units.append(np.full(valid_length, unit))

    return (
        np.concatenate(truncated_residuals),
        np.concatenate(truncated_ruls),
        np.concatenate(truncated_cycles),
        np.concatenate(truncated_units)
    )

# Apply truncation to training and testing data
residuals_train_data, ruls_train_data, cycles_train_data, units_train_data = truncate_data_by_rul(
    residuals_train_data, ruls_train_data, cycles_train_data, units_train_data
)
residuals_test_data, ruls_test_data, cycles_test_data, units_test_data = truncate_data_by_rul(
    residuals_test_data, ruls_test_data, cycles_test_data, units_test_data
)

X_train = residuals_train_data
y_train = ruls_train_data

X_test = residuals_test_data
y_test = ruls_test_data

# Step 2: Define dataset and dataloader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Step 3: Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        # First layer: Linear transformation + ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # Second layer: Linear transformation + ReLU
        x = self.fc2(x)
        x = self.relu(x)

        # Third layer: Linear transformation + ReLU
        x = self.fc3(x)
        x = self.relu(x)

        # Fourth layer: Linear transformation + ReLU
        x = self.fc4(x)
        x = self.relu(x)

        # Fifth layer: Linear transformation + ReLU
        x = self.fc5(x)
        x = self.relu(x)

        # Output layer: Linear transformation to single output
        x = self.fc6(x)
        return x

# Step 5: Initialize model, loss, and optimizer
input_size = X_train.shape[1]
model = SimpleNN(input_size)
criterion = nn.SmoothL1Loss() #Huber loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, verbose=True)

# Step 6: Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Average training loss
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Step 7: Evaluate and plot results
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    y_pred_actual = y_pred
    y_actual = y_test

# Calculate RMSE (root mean square error)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
print(f"RMSE: {rmse:.4f}")

# Directory to save plots
output_dir = "plots/RUL_per_unit"
os.makedirs(output_dir, exist_ok=True)

# Plot predicted vs actual RUL per unit
unique_units = np.unique(units_test_data)

for unit in unique_units:
    unit_indices = units_test_data == unit
    cycles_unit = cycles_test_data[unit_indices]
    actual_rul_unit = y_actual[unit_indices]
    predicted_rul_unit = y_pred_actual[unit_indices]

    plt.figure(figsize=(8, 6))
    plt.plot(cycles_unit, actual_rul_unit, label='Actual RUL', color='blue', marker='o')
    plt.plot(cycles_unit, predicted_rul_unit, label='Predicted RUL', color='orange', marker='x')
    plt.title(f"Unit {int(unit)}: Predicted vs Actual RUL")
    plt.xlabel('Cycle')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid()

    file_path = os.path.join(output_dir, f"RUL_unit_{int(unit)}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
