import numpy as np
import torch

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


import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.fft import fft2, ifft2, fftshift

def apply_filters(data, filter_type="gaussian", kernel_size=3, sigma=1.0, cutoff_frequency=0.1):
    """
    Applique différents types de filtrage pour réduire le bruit sur les données d'entrée.

    Arguments :
    - data : np.array de taille (N, C, H, W), les images d'entrée
    - filter_type : str, le type de filtre à appliquer ("gaussian", "median", "mean", "fft_lowpass")
    - kernel_size : int, taille du noyau pour les filtres médian et moyenneur
    - sigma : float, écart-type pour le filtre gaussien
    - cutoff_frequency : float, fréquence de coupure pour le filtre passe-bas FFT
    
    Retourne :
    - data_filtered : np.array, les données après filtrage, normalisées entre 0 et 1
    """
    data_filtered = np.zeros_like(data)
    
    for i in range(data.shape[0]):  # Boucle sur chaque image
        image = data[i, 0]  # Supposons que les données soient en (N, C, H, W)

        if filter_type == "gaussian":
            # Appliquer un filtre gaussien
            filtered_image = gaussian_filter(image, sigma=sigma)

        elif filter_type == "median":
            # Appliquer un filtre médian
            filtered_image = median_filter(image, size=kernel_size)

        elif filter_type == "mean":
            # Appliquer un filtre moyenneur (box blur)
            filtered_image = uniform_filter(image, size=kernel_size)

        elif filter_type == "fft_lowpass":
            # Appliquer un filtre passe-bas FFT
            F = fft2(image)
            Fshift = fftshift(F)

            # Calculer le masque passe-bas
            rows, cols = image.shape
            crow, ccol = rows // 2 , cols // 2
            mask = np.zeros((rows, cols))
            radius = int(cutoff_frequency * min(rows, cols))
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
            mask[mask_area] = 1

            # Appliquer le masque et réaliser la transformation inverse
            Fshift_filtered = Fshift * mask
            F_ishift = np.fft.ifftshift(Fshift_filtered)
            filtered_image = np.abs(ifft2(F_ishift))
        
        else:
            raise ValueError(f"Filtre {filter_type} non supporté")

        # Normaliser pour garder les valeurs entre 0 et 1
        filtered_image = np.clip(filtered_image, 0, 1)
        data_filtered[i, 0] = filtered_image  # Remettre dans le tableau de sortie
    
    return data_filtered



import matplotlib.pyplot as plt

def plot_velocity_comparison(u_true, v_true, u_pred, v_pred, filename, test_1, test_2, test_code, coord1=(60, 60), coord2=(100, 100)):
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
    axs[0].plot(u_true[:, coord1[0], coord1[1]], label=f"{test_1} (u) {coord1}")
    axs[0].plot(u_pred[:, coord1[0], coord1[1]], label=f"{test_2} (u) {coord1}")
    #axs[0].plot(u_true[:, coord2[0], coord2[1]], label=f"True (u) {coord2}")
    #axs[0].plot(u_pred[:, coord2[0], coord2[1]], '--', label=f"Predicted (u) {coord2}")
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('u (x, y) m/s')
    axs[0].set_title(f'Velocity u(x,y,t) at coordinates {coord1}')
    axs[0].grid(True)
    axs[0].legend()

    # Velocity v component plot
    axs[1].plot(v_true[:, coord1[0], coord1[1]], label=f"{test_1} (v) {coord1}")
    axs[1].plot(v_pred[:, coord1[0], coord1[1]], label=f"{test_2} (v) {coord1}")
    #axs[1].plot(v_true[:, coord2[0], coord2[1]], label=f"True (v) {coord2}")
    #axs[1].plot(v_pred[:, coord2[0], coord2[1]], '--', label=f"Predicted (v) {coord2}")
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('v (x, y) m/s')
    axs[1].set_title(f'Velocity v(x,y,t) at coordinates {coord1}')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"{test_code}/{filename}.png")
    #plt.show()
    
    
def plot_filtering_comparison(u_true, u_pred, test_code, test_1='Noisy', test_2='After Filtering', coord1=(60, 60), coord2=(100, 100)):
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
    #, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Velocity u component plot
    plt.plot(u_true[:100, coord1[0], coord1[1]], label=f"{test_1} (Luminosity) {coord1}")
    plt.plot(u_pred[:100, coord1[0], coord1[1]], label=f"{test_2} (Luminosity) {coord1}")
    plt.xlabel('Time (s)')
    plt.ylabel('Luminosity')
    plt.title(f'Particles Luminosity at coordinates {coord1}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{test_code}/Filtering_Comparison.png")
    #plt.show()
    
def visualize_imgs_differences(img_1, img_2, img_3, test_code, test_1='H015', test_2='H016', test_3='H017', coord1=(128, 128), coord2=(100, 100)):
    #, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Velocity u component plot
    plt.plot(img_1[:100, coord1[0], coord1[1]], label=f"{test_1} (Luminosity) {coord1}")
    plt.plot(img_2[:100, coord1[0], coord1[1]], label=f"{test_2} (Luminosity) {coord1}")
    plt.plot(img_3[:100, coord1[0], coord1[1]], label=f"{test_3} (Luminosity) {coord1}")
    plt.xlabel('Time (s)')
    plt.ylabel('Luminosity')
    plt.title(f'Particles Luminosity at coordinates {coord1}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{test_code}/Comparison_Images.png")
    #plt.show()