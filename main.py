import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

from models_2d import UNetLSTM2D
from utilis import * 
from sp_functions import *
from plotting import *


###############################################################
# Configuration and Hyperparameters
###############################################################
start_time = time.time()

# Input image dimensions
input_height = 256
input_width = 256

num_epochs = 600
batch_size = 32
base_pred_path = "/notebooks/aPivMLSP/Final_tests"
pred_code = "12_Dec_TC_01_Method_06"
pred_path = f"{base_pred_path}/{pred_code}/model_UnetLSTM2D_saved_{pred_code}.pth"
step_size = 300
learning_rate = 1e-2

denoising_method = "05"
"""
01 : Gaussian
02 : Median
O3 : Non-Local Means (NLM)
04 : Total Variation
05 : Anisotropic Diffusion
"""


mode = "Train"
test_code = f"18_Dec_TC_01_Method_{denoising_method}"
USE_SEQ_IMG = False  # False = use Pair Images
start_dim = 8

if USE_SEQ_IMG:
    input_dim = 1
else:
    input_dim = 2

sequence_length = 1

directory_path = os.path.join('Final_tests', test_code)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)



###############################################################
# Data Loading
###############################################################
targets_data_H17 = np.load("/notebooks/clean_data_H017/data_ground_truth_H017.npy")
inputs_data_H17 = np.load("/notebooks/clean_data_H017/noisy_images_data_H017.npy")


# Create consecutive pairs for each dataset and then concatenate them
inputs_data, targets_data = create_consecutive_pairs_v2(inputs_data_H17, targets_data_H17)

# Normalize input data
max_value = inputs_data.max()
inputs_data = inputs_data / max_value

print(f"Inputs shape: {inputs_data.shape}")
print(f"Targets shape: {targets_data.shape}")

# Denoise the PIV images using a chosen method
if denoising_method == "01":
    inputs_data = denoise_piv_images(inputs_data, method='gaussian', sigma=1.5)
    #print("Max value after denoising:", inputs_data.max())
elif denoising_method == "02":
    inputs_data = denoise_piv_images(inputs_data, method='median', size=5)
elif denoising_method == "03":
    inputs_data = denoise_piv_images(inputs_data, method='non_local_means', h=1.0, fast_mode=True)
elif denoising_method == "04":
    inputs_data = denoise_piv_images(inputs_data, method='tv', weight=0.1)
elif denoising_method == "05":
    inputs_data = denoise_piv_images(inputs_data, method='anisotropic_diffusion', iterations=10, kappa=50, step=0.1)
elif denoising_method == "06":
    inputs_data = denoise_piv_images(inputs_data, method='bm3d', sigma_psd=1.5)
else:
    pass

###############################################################
# Data Preparation for Training/Validation/Testing
###############################################################
def permute_dimensions(tensor):
    """Permute dimensions if single sequence images are used."""
    return tensor.transpose(0, 2, 1, 3, 4)

X = inputs_data
Y = targets_data

if input_dim == 1:
    X = permute_dimensions(X)

total_steps = X.shape[0]

# Train/val/test split (80/5/15)
train_split = int(0.8 * len(X))
val_split = int(0.85 * len(X))  # 5% validation
X_train, Y_train = X[:train_split], Y[:train_split]
X_val, Y_val = X[train_split:val_split], Y[train_split:val_split]
X_test, Y_test = X[val_split:], Y[val_split:]

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

print(f"X_train shape: {X_train_tensor.shape}")
print(f"X_test shape: {X_test_tensor.shape}")
print(f"Y_train shape: {Y_train_tensor.shape}")
print(f"Y_test shape: {Y_test_tensor.shape}")

###############################################################
# Model, Optimizer, and Loss Definition
###############################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNetLSTM2D(input_dim=input_dim, num_classes=2, start_dim=start_dim, 
                   input_height=input_height, input_width=input_width).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
criterion = torch.nn.MSELoss()

# Dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

###############################################################
# Training
###############################################################
train_losses = []
val_losses = []
train_rmses = []
val_rmses = []

if mode == "Train":
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_rmse_sum = 0
        train_rel_error_sum = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            with torch.no_grad():
                train_rmse, train_rel_err = metrics(output, target)
                train_rmse_sum += train_rmse
                train_rel_error_sum += train_rel_err

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_rmse = train_rmse_sum / len(train_loader)
        avg_train_rel_error = train_rel_error_sum / len(train_loader)

        val_loss, val_rmse, val_rel_err = eval_model(model, criterion, test_loader, device)
        scheduler.step()

        print(f'Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]}')

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_rmses.append(avg_train_rmse)
        val_rmses.append(val_rmse)

        # Early stopping condition example:
        if val_loss <= 5e-4 and avg_train_loss <= 9e-4:
            break

    # Save the trained model
    model_file_path = os.path.join(directory_path, f'model_UnetLSTM2D_saved_{test_code}.pth')
    torch.save(model.state_dict(), model_file_path)

    # Plot training curves
    train_rmses = torch.stack(train_rmses).cpu().numpy()
    val_rmses = torch.stack(val_rmses).cpu().numpy()
    loss_pic_path = os.path.join(directory_path, f'loss_{test_code}.png')
    plot_loss_rmse(train_losses, val_losses, train_rmses, val_rmses, loss_pic_path)
else:
    # If not training, load a pre-trained model
    print(f"Loading model from {pred_path}")
    model.load_state_dict(torch.load(pred_path, map_location=device))
    model.to(device)

###############################################################
# Evaluation on Test Set
###############################################################
model.eval()
test_predictions = predict(model, test_loader, device)

# Create directories for saving predictions
directory_picU = os.path.join(f'tests/{test_code}', "pics_U")
if not os.path.exists(directory_picU):
    os.makedirs(directory_picU)

directory_picV = os.path.join(f'tests/{test_code}', "pics_V")
if not os.path.exists(directory_picV):
    os.makedirs(directory_picV)

u_pred = test_predictions[:, 0]
v_pred = test_predictions[:, 1]

u_true = Y_test_tensor.numpy()[:, 0]
v_true = Y_test_tensor.numpy()[:, 1]



min_timesteps = min(u_true.shape[0], u_pred.shape[0])

# Compute error metrics over time steps
E_u, E_v, _ = zip(*[compute_metric_E(u_true[i], u_pred[i], v_true[i], v_pred[i], v_true[i], v_pred[i], mode="min")
                    for i in range(min_timesteps)])
Emae_u = [compute_metric_MAEt(u_true[i], u_pred[i]) for i in range(min_timesteps)]
Emae_v = [compute_metric_MAEt(v_true[i], v_pred[i]) for i in range(min_timesteps)]

end_time = time.time()
execution_time = end_time - start_time
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = execution_time % 60

# Select some timesteps for visualization
select_times = [10, 20, 30] 
error_u = u_pred - u_true
error_v = v_pred - v_true

for i in select_times:
    print(f"Visualizing step {i}")
    plot_side_by_side(u_true[i], u_pred[i], error_u[i], i, "U", directory_picU, test_code, shading="flat")
    plot_side_by_side(v_true[i], v_pred[i], error_v[i], i, "V", directory_picV, test_code, shading="flat")

rmse = compute_rmse(u_pred, u_true, v_pred, v_true)
aee = compute_aee(u_pred, u_true, v_pred, v_true)

print(f"RMSE: {rmse}")
print(f"AEE: {aee}")

# Plot error metrics
plot_error_metric(E_u, E_v, directory_path, test_code)
plot_mae_metric(Emae_u, Emae_v, directory_path, test_code)

print(f"Execution time: {hours}h:{minutes}m:{seconds:.2f}s")

###############################################################
# Save Results to File
###############################################################
results_path = os.path.join(directory_path, f'Results_{test_code}.txt')
with open(results_path, "w") as f:
    f.write(f"**** TEST CONFIG ({test_code}) ****\n")
    f.write(f"Sequence Length: {sequence_length}\n")
    f.write(f"Batch size: {batch_size}\n\n")
    f.write(f"**** TRAINING CONFIG ****\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Epochs: {num_epochs}\n\n")
    f.write(f"**** RESULTS ****\n")
    f.write(f"Execution time: {hours}h:{minutes}m:{seconds:.2f}s\n")
    f.write(f"Average RMSE: {rmse}\n")
    f.write(f"Max AEE: {aee}\n")
