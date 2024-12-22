import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

###############################################################
# Utility Functions
###############################################################
def compute_rmse(u_pred, u_true, v_pred, v_true):
    """Compute the root mean squared error (RMSE) for velocity predictions."""
    rmse = np.sqrt(np.mean((u_pred - u_true)**2 + (v_pred - v_true)**2))
    return rmse

def compute_aee(u_pred, u_true, v_pred, v_true):
    """Compute the average endpoint error (AEE) for velocity predictions."""
    ee = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
    aee = np.mean(ee)
    return aee

def create_consecutive_pairs_v2(inputs_data, targets_data, output_size=(256, 256)):
    """
    Create consecutive pairs of input images along with their corresponding targets.
    This function resizes both inputs and targets to the specified output size.
    """
    # Create consecutive pairs of input images
    inputs_pairs = [(inputs_data[i], inputs_data[i + 1]) for i in range(len(inputs_data) - 1)]
    inputs_pairs = np.array(inputs_pairs)[:, np.newaxis, :, :, :]  # (N-1, 2, H, W)

    # Resize input pairs
    inputs_resized = np.array([
        [
            cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
            for img in pair[0]
        ]
        for pair in inputs_pairs
    ])

    # Resize corresponding targets
    targets_resized = np.array([
        [
            cv2.resize(targets_data[i, j], output_size, interpolation=cv2.INTER_LINEAR)
            for j in range(targets_data.shape[1])
        ]
        for i in range(1, len(targets_data))
    ])

    # Return arrays of shape: inputs (N-1, 2, H, W), targets (N-1, 2, H, W)
    return inputs_resized[:, np.newaxis, :, :, :], targets_resized


def compute_metric_E(u_true, u_pred, v_true, v_pred, p_true, p_pred, mode="min"):
    #print(f"u_true shape: {true.shape}, u_pred shape: {pred.shape}")
    if mode == "max":
        normalizer = np.max([np.sum(u_true ** 2), np.sum(v_true ** 2), np.sum(p_true ** 2)])
    else:
        normalizer = np.min([np.sum(u_true ** 2), np.sum(v_true ** 2), np.sum(p_true ** 2)])
    et_u = np.sum((u_true - u_pred) ** 2) / normalizer
    et_v = np.sum((v_true - v_pred) ** 2) / normalizer
    #et_w = np.sum((w_true - w_pred) ** 2) / normalizer
    et_p = np.sum((p_true - p_pred) ** 2) / normalizer

    return et_u, et_v, et_p

def compute_metric_MAEt(true, pred):
    #print(f"u_true shape: {true.shape}, u_pred shape: {pred.shape}")

    return np.mean(np.abs(true - pred))

def metrics(predictions, targets):
    if not (isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor)):
        raise TypeError("Les prédictions et les cibles doivent être des tenseurs PyTorch.")
    
    # Compute RMSE value
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    
    # Compute Relative Error L2 in %
    relative_l2_error = torch.norm(predictions - targets) / torch.norm(targets)
    relative_l2_error_percentage = 100 * relative_l2_error

    return rmse, relative_l2_error_percentage

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def predict(model, data_loader, device):
    model.eval()  # Mode d'évaluation
    predictions = []
    with torch.no_grad():  # Pas de calcul de gradient
        for batch_x, batch_y in data_loader:
            inputs = batch_x.to(device)
            outputs = model(inputs)
            predictions.append(outputs.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def eval_model(model, criterion, test_loader, device):
    model.eval()
    val_loss_sum = 0
    val_rmse_sum, val_relative_l2_error_sum = 0, 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            val_loss_sum += loss.item()
            val_rmse, val_relative_l2_error = metrics(output, target)
            val_rmse_sum += val_rmse
            val_relative_l2_error_sum += val_relative_l2_error
            
    val_loss = val_loss_sum / len(test_loader)
    avg_val_rmse = val_rmse_sum / len(test_loader)
    avg_val_relative_l2_error = val_relative_l2_error_sum / len(test_loader)
    return val_loss, avg_val_rmse, avg_val_relative_l2_error


# Fonctions de normalisation
def calculate_stats(data):
    return np.mean(data, axis=(0, 2, 3), keepdims=True), np.std(data, axis=(0, 2, 3), keepdims=True)

def normalize(data, mean, std):
    return (data - mean) / std

def inverse_normalize(data_normalized, mean, std):
    return data_normalized * std + mean

# Fonctions de normalisation Min-Max
def calculate_min_max(data):
    min_val = np.min(data, axis=(0, 2, 3), keepdims=True)
    max_val = np.max(data, axis=(0, 2, 3), keepdims=True)
    return min_val, max_val

def normalize_min_max(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def inverse_normalize_min_max(data_normalized, min_val, max_val):
    return data_normalized * (max_val - min_val) + min_val


def create_sequences(total_steps, sequence_length, inputs, targets, spatial_size):
    # Calcul du nombre de séquences
    num_sequences = total_steps - sequence_length

    # Pré-allocation de la mémoire pour X et Y
    X = np.empty((num_sequences, sequence_length, inputs.shape[1], *spatial_size), dtype=np.float32)
    Y = np.empty((num_sequences, targets.shape[1], *spatial_size), dtype=np.float32)

    print("Creating Sequences")
    # Remplissage des séquences
    for i in range(num_sequences):
        X[i] = inputs[i:i + sequence_length]  # Remplissage de X avec la séquence
        Y[i] = targets[i + sequence_length]    # Remplissage de Y avec la séquence cible
    return X, Y


def plot_velocity_comparison(u_true, v_true, u_pred, v_pred, test_code, filename="Model_Evaluation", test_1="True", test_2="Pred.", coord1=(128, 128), coord2=(250, 250)):
    """
    Plots a comparison of true vs. predicted velocity components at specific coordinates.

    Args:
        u_true (np.array): True u velocity component, shape (time, height, width).
        v_true (np.array): True v velocity component, shape (time, height, width).
        u_pred (np.array): Predicted u velocity component, shape (time, height, width).
        v_pred (np.array): Predicted v velocity component, shape (time, height, width).
        coord1 (tuple): First coordinate (x, y) for comparison.
        coord2 (tuple): Second coordinate (x, y) for comparison.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Velocity u component plot
    axs[0].plot(u_true[:, coord1[0], coord1[1]], label=f"{test_1} (u) {coord1}", color='green')
    axs[0].plot(u_pred[:, coord1[0], coord1[1]], label=f"{test_2} (u) {coord1}", color='red')
    
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('u (t) m/s')
    axs[0].set_title(f'Velocity u(x,y,t) at pixel-coordinates {coord1}')
    axs[0].grid(True)
    axs[0].legend()

    # Velocity v component plot
    axs[1].plot(v_true[:, coord1[0], coord1[1]], label=f"{test_1} (v) {coord1}", color='green')
    axs[1].plot(v_pred[:, coord1[0], coord1[1]], label=f"{test_2} (v) {coord1}", color='red')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('v (t) m/s')
    axs[1].set_title(f'Velocity v(x,y,t) at pixel-coordinates {coord1}')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"{test_code}/{filename}.png")
    
    
    
    
    

    



