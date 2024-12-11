import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from matplotlib import patches, tri
from sklearn.preprocessing import StandardScaler
#from models_original import *
#from models_sanslstm import *
from models_2d import *
import torch.optim as optim
from scipy.interpolate import griddata
import pandas as pd
from plotting import *
import os
import time
from utilis import *
from sp_functions import *


# Start recording time
start_time = time.time()

input_height = 256
input_width = 256

num_epochs = 100 #0 #00
batch_size = 32 # Taille du batch ajustable selon la mémoire GPU disponible
base_pred_path = "/notebooks/aPivMLSP/tests"
pred_code = "07_Novembre_A24_TC_01_H017"
pred_path = f"{base_pred_path}/{pred_code}/model_UnetLSTM2D_saved_{pred_code}.pth"
step_size = 50
learning_rate = 1e-2

mode = "Train"
test_code = "30_Nov_TC_04_H017_Method_06_REF"
USE_SEQ_IMG = False # False = use Pair Images
start_dim = 8

if USE_SEQ_IMG:
    input_dim = 1
else:
    input_dim = 2
#test_code = "07_Novembre_A24_TC_01_H017_Eval_Soft"
#test_code = "07_Novembre_A24_TC_01_H017_Eval_Hard_SQG"

# Création des séquences
sequence_length = 1

# Creating the directory if it doesn't exist
directory_path = os.path.join('tests', test_code)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


# Define functions to compute RMSE and AEE
def compute_rmse(u_pred, u_true, v_pred, v_true):
    rmse = np.sqrt(np.mean((u_pred - u_true)**2 + (v_pred - v_true)**2))
    return rmse

def compute_aee(u_pred, u_true, v_pred, v_true):
    ee = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
    aee = np.mean(ee)
    return aee
    
    
targets_data_H17 = np.load("/notebooks/clean_data_H017/data_ground_truth_H017.npy")
inputs_data_H17 = np.load("/notebooks/clean_data_H017/noisy_images_data_H017.npy")

targets_data_H16 = np.load("/notebooks/clean_data_H017/data_ground_truth_H016.npy")
inputs_data_H16 = np.load("/notebooks/clean_data_H017/noisy_images_data_H016.npy")

#inputs_data = np.concatenate((inputs_data_H16, inputs_data_H17), axis=0)
#targets_data = np.concatenate((targets_data_H16, targets_data_H17), axis=0)


# Denoising Techniques
#inputs_data = denoise_wavelet(inputs_data)
#inputs_data = gaussian_filter(inputs_data, sigma=1.0)


inputs_data_SQG = np.load("/notebooks/clean_data_H017/images_data_SQG.npy")
targets_data_SQG = np.load("/notebooks/clean_data_H017/flows_data_SQG.npy")

inputs_data_SQG = np.transpose(inputs_data_SQG, (0, 3, 1, 2))[:, np.newaxis, :, :, :]  # (1500, 2, 1, 256, 256)
# Reshape des cibles
targets_data_SQG = np.transpose(targets_data_SQG, (0, 3, 1, 2))  # (1500, 2, 256, 256)
#print(f"Inputs SQG shape : {inputs_data_SQG.shape}")
#print(f"Targets SQG shape : {targets_data_SQG.shape}")



#inputs_data = inputs_data[:498]
#targets_data = targets_data[:498]

import numpy as np
import cv2  # Utilisé pour la redimension via interpolation (OpenCV)

def create_consecutive_pairs_v2(inputs_data, targets_data, output_size=(256, 256)):
    # Crée des paires consécutives (img1, img2), (img2, img3), ..., (img497, img498)
    inputs_pairs = [(inputs_data[i], inputs_data[i + 1]) for i in range(len(inputs_data) - 1)]
    
    # Convertir en numpy array et ajouter une dimension pour obtenir (497, 2, 120, 120)
    inputs_pairs = np.array(inputs_pairs)[:, np.newaxis, :, :, :]  # (497, 2, 1, 120, 120)
    
    # Redimensionner les paires d'entrées à (256, 256)
    inputs_resized = np.array([
        [
            cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR) 
            for img in pair[0]
        ]
        for pair in inputs_pairs
    ])
    
    # Redimensionner les cibles (targets_data[1:]) à (256, 256)
    targets_resized = np.array([
        [
            cv2.resize(targets_data[i, j], output_size, interpolation=cv2.INTER_LINEAR)
            for j in range(targets_data.shape[1])
        ]
        for i in range(1, len(targets_data))
    ])  # (48, 2, 256, 256)
    
    return inputs_resized[:, np.newaxis, :, :, :], targets_resized   # Résultat avec la taille (497, 2, 256, 256)

# Générer les paires pour images
def create_consecutive_pairs(inputs_data, targets_data):
    # Crée des paires consécutives (img1, img2), (img2, img3), ..., (img497, img498)
    inputs_data = [(inputs_data[i], inputs_data[i + 1]) for i in range(len(inputs_data) - 1)]
    #inputs_data = inputs_data[:, np.newaxis, :, :, :]   # (1500, 2, 1, 256, 256)
    return np.array(inputs_data)[:, np.newaxis, :, :, :], targets_data[1:]   # Résultat de forme (497, 2, 120, 120)

# Réorganiser data_images en commençant par data2, data3, ..., data498
def skip_first(data_images):
    # Retourner les données à partir du deuxième élément
    return data_images[1:]  # Résultat de forme (497, 2, 120, 120)

#inputs_data = create_consecutive_pairs(inputs_data)  # (497, 2, 120, 120)
#targets_data = skip_first(targets_data)  # (497, 2, 120, 120)
inputs_data, targets_data = create_consecutive_pairs_v2(inputs_data_H17,targets_data_H17)

#inputs_data, targets_data = inputs_data_H17, targets_data_H17


#visualize_imgs_differences(img_H15, img_H16, img_H17, directory_path)
#inputs_data = np.transpose(inputs_data, (0, 3, 1, 2))[:, np.newaxis, :, :, :]  # (1500, 2, 1, 256, 256)
# Reshape des cibles
#targets_data = np.transpose(targets_data, (0, 3, 1, 2))  # (1500, 2, 256, 256)



max_value = inputs_data.max()
#inputs_data = inputs_data / max_value


scaler = StandardScaler()
original_shape = inputs_data.shape

print(f"Inputs shape : {inputs_data.shape}")
print(f"Targets shape : {targets_data.shape}")

# method 01
#inputs_data = denoise_piv_images(inputs_data, method='gaussian', sigma=1.5)
#inputs_data = denoise_piv_images(inputs_data, method='median', size=5)
#inputs_data = denoise_piv_images(inputs_data, method='wavelet', wavelet='db1', mode='hard')
#inputs_data = denoise_piv_images(inputs_data, method='non_local_means', h=1.0, fast_mode=True)
#inputs_data = denoise_piv_images(inputs_data, method='fft', threshold=0.1)


#inputs_data = denoise_piv_images(inputs_data, method='bilateral', sigma_color=0.05, sigma_spatial=15)



#print(original_shape)
#data_reshaped = inputs_data.reshape(-1, original_shape[-1])  # Aplatit les dimensions supérieures
#print(data_reshaped.shape)

#data_scaled = scaler.fit_transform(data_reshaped)
#inputs_data = data_scaled.reshape(original_shape)

#inputs_data = scaler.fit_transform(data_scaled)
        
#print("Valeur maximale :", max_value)
print("Valeur maximale :", inputs_data.max())

total_steps = 497
#num_features = 2  # u, v, w, p
spatial_size = (256, 256)  # Taille des dimensions spatiales (96x96x96)

# Calcul du nombre de séquences
num_sequences = total_steps - sequence_length
def permute_dimensions(tensor):
    return tensor.transpose(0, 2, 1, 3, 4)

X = inputs_data #[:total_steps]
Y = targets_data #[:total_steps]
# Combinaison des données dans un seul array pour facilité
#X, Y = create_sequences(total_steps, sequence_length, inputs_data, targets_data, spatial_size)
#X, Y = create_sequences(total_steps, sequence_length, targets_data, targets_data, spatial_size)
if input_dim==1:
    X = permute_dimensions(X)
# Split en ensembles de données
train_split = int(0.8*len(X))
val_split = int(0.8*len(X))
end_split = int(0.9*len(X))

X_train, Y_train = X[:train_split], Y[:train_split]
X_val, Y_val = X[val_split:end_split], Y[val_split:end_split]
X_test, Y_test = X[395:], Y[395:]


#X_train = np.concatenate((inputs_data_SQG_train, X_train), axis=0)
#Y_train = np.concatenate((targets_data_SQG_train, Y_train), axis=0)

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

#X_train = X_train.astype(np.float32)
#X_train = X_train.astype(np.float32)
#X_train = X_train.astype(np.float32)

# Conversion en tensors PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)


print(f"Taille XTrain: {X_train_tensor.shape}")
print(f"Taille XTest: {X_test_tensor.shape}")

print(f"Taille YTrain: {Y_train_tensor.shape}")
print(f"Taille YTest: {Y_test_tensor.shape}")

# Initialisation du modèle
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model =  UNetLSTM2D(input_dim=input_dim, num_classes=2, start_dim=start_dim, input_height=input_height, input_width=input_width).to(device)

#model =  UNet2D(num_classes=2, start_dim=start_dim, input_height=input_height, input_width=input_width).to(device)

#model = UNet().to(device)

# Définition de l'optimiseur et de la fonction de perte
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)  #optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-5, verbose=True)
criterion = torch.nn.MSELoss()


# DataLoader pour l'entraînement
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(X_test_tensor, Y_test_tensor) # Assurez-vous que X_test_tensor est correctement préparé
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


train_losses = []
val_losses = []
train_rmses = []
val_rmses = []

if mode == "Train":
    # Entraînement du modèle
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_loss_sum = 0
        train_rmse_sum, train_relative_l2_error_sum = 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            with torch.no_grad():
                train_rmse, train_relative_l2_error = metrics(output, target)
                train_rmse_sum += train_rmse 
                train_relative_l2_error_sum += train_relative_l2_error
            #print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        average_loss = total_loss / len(train_loader)
        avg_train_rmse = train_rmse_sum / len(train_loader)
        avg_train_relative_l2_error = train_relative_l2_error_sum / len(train_loader)
        val_loss, val_rmse, val_rel_err = eval_model(model, criterion, test_loader, device)

        if val_loss <= 5e-4 and average_loss <= 9e-4:
            break
        #scheduler.step(average_loss)
        scheduler.step()
        print(f'Epoch {epoch}/{num_epochs}, Train Loss: {average_loss}, Val Loss: {val_loss}, lr: {scheduler.get_last_lr()[0]}')
        train_losses.append(average_loss)
        val_losses.append(val_loss)
        train_rmses.append(avg_train_rmse)
        val_rmses.append(val_rmse)


    model_file_path = os.path.join(directory_path, f'model_UnetLSTM2D_saved_{test_code}.pth') #"model_unet_convLSTM.pth"
    torch.save(model.state_dict(), model_file_path)

    train_rmses = torch.stack(train_rmses)
    val_rmses = torch.stack(val_rmses)
    train_rmses = train_rmses.cpu().numpy()
    val_rmses = val_rmses.cpu().numpy()
    loss_pic_path = os.path.join(directory_path, f'loss_{test_code}.png')
    # Plot losses and RMSEs
    plot_loss_rmse(train_losses, val_losses, train_rmses, val_rmses, loss_pic_path)
    # Évaluation sur l'ensemble de test

else:
    # Évaluation sur l'ensemble de test
    print(f"Loading the model from {pred_path}")
    model.load_state_dict(torch.load(pred_path, map_location=device))
    model.to(device)

model.eval()
# Faire les prédictions
test_predictions = predict(model, test_loader, device)

# Creating the directory if it doesn't exist
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

plot_velocity_comparison(u_true, v_true, u_pred, v_pred, directory_path)
plot_velocity_comparison_max(u_true, v_true, u_pred, v_pred, directory_path)
                         
min_timesteps = min(u_true.shape[0], u_pred.shape[0])

E_u, E_v, E_p = zip(*[compute_metric_E(u_true[i], u_pred[i], v_true[i], v_pred[i], v_true[i], v_pred[i], mode="min") for i in range(min_timesteps)])

# Calcul de la métrique E(t) pour chaque variable (u, v, w, p)
Emae_u = [compute_metric_MAEt(u_true[i, :, :], u_pred[i, :, :]) for i in range(min_timesteps)]
Emae_v = [compute_metric_MAEt(v_true[i, :, :], v_pred[i, :, :]) for i in range(min_timesteps)]

end_time = time.time()
execution_time = end_time - start_time
# Convertir en heures, minutes, secondes
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = execution_time % 60

max_errors = []
x_min, x_max, y_min, y_max = 0, 6, 0, 2
select_times = [10, 20, 30, 40, 49, 60, 70, 80, 100]

for i in select_times:
    print(f"Time Pred -- @ {i} step")
    error_u = u_pred - u_true
    error_v = v_pred - v_true
    #error_p = p_pred - p_true
    
    max_errors.append(np.abs(np.mean(error_u)))
    max_errors.append(np.abs(np.mean(error_v)))
    #max_errors.append(np.abs(np.mean(error_p)))
    plot_side_by_side(u_true[i], u_pred[i], error_u[i], i+1, "U", directory_picU, test_code, shading="flat")
    plot_side_by_side(v_true[i], v_pred[i], error_v[i], i+1, "V", directory_picV, test_code, shading="flat")
    #plot_side_by_side(p_true[i], p_pred[i], error_p[i], i, "P", directory_picP, test_code, x_min, x_max, y_min, y_max, shading="flat")
            
rmse = compute_rmse(u_pred, u_true, v_pred, v_true)
aee = compute_aee(u_pred, u_true, v_pred, v_true)
average_max_error = rmse #np.mean(max_errors)
FINAL_max_error = aee #np.max(max_errors)

print(f"MOY des erreurs : {average_max_error}")
print(f"MAX des erreurs : {FINAL_max_error}")
plot_error_metric(E_u, E_v, directory_path, test_code)
plot_mae_metric(Emae_u, Emae_v, directory_path, test_code)

print(f"Temps : {hours} heures, {minutes} min et {seconds:.2f} second \n")
# SAVING RESULTS
results_path = os.path.join(directory_path, f'Results_{test_code}.txt')


# Écrire dans un fichier
with open(results_path, "w") as f:
    f.write(f"**** CONFIG. DU TEST ({test_code}) **** \n")
    #f.write(f"Desription du test: {test_desc} \n")
    #f.write(f"Params Réseau: {num_layers}x{hidden_dim} \n")
    f.write(f"Sequence Length: {sequence_length} \n")
    f.write(f"Bacht size: {batch_size} \n")
    f.write(f"\n")
    f.write(f"**** CONFIG. TRAINING **** \n")
    #f.write(f"Optimizer: {choice_optim} \n")
    f.write(f"Learning Rate: {learning_rate} \n")
    #f.write(f"weight_decay: {weight_decay} \n")
    f.write(f"Epochs: {num_epochs} \n")
    f.write(f"\n")
    f.write(f"**** RESULTS **** \n")
    f.write(f"Temps : {hours} heures, {minutes} min et {seconds:.2f} second \n")
    f.write(f"MOY des erreurs : {average_max_error} \n")
    f.write(f" \n")

