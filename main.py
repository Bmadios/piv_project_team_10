import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from matplotlib import patches, tri

from models_2d import *
import torch.optim as optim
from scipy.interpolate import griddata
import pandas as pd
from plotting import *
import os
import time
from utilis import *

# Start recording time
start_time = time.time()

input_height = 256
input_width = 256

num_epochs = 100 #0 #00
batch_size = 16 # Taille du batch ajustable selon la mémoire GPU disponible
base_pred_path = "/notebooks/lid/model_training/tests"
pred_code = "06_Octobre_A24_TC_02"
pred_path = f"{base_pred_path}/{pred_code}/model_unet_convLSTM.pth"
step_size = 50
learning_rate = 1e-2

mode = "Train"
test_code = "06_Novembre_A24_TC_05_SQG"
start_dim = 32
# Création des séquences
sequence_length = 2

# Creating the directory if it doesn't exist
directory_path = os.path.join('tests', test_code)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

arg_P1 = "H015"
arg_P2 = "H016"
arg_P3 = "H017"
#targets_data = np.load("/notebooks/clean_data_H017/data_ground_truth_H017.npy")
#inputs_data = np.load("/notebooks/clean_data_H017/noisy_images_data_H017.npy")


#targets_data = np.load("/notebooks/clean_data_H017/data_ground_truth_256x256_H015.npy")
#inputs_data = np.load("/notebooks/clean_data_H017/noisy_images_data_256x256_H015.npy")

img_H15 = np.load(f"/notebooks/clean_data_H017/noisy_images_data_Padding_{arg_P1}.npy")
img_H16 = np.load(f"/notebooks/clean_data_H017/noisy_images_data_Padding_{arg_P2}.npy")
img_H17 = np.load(f"/notebooks/clean_data_H017/noisy_images_data_Padding_{arg_P3}.npy")

visualize_imgs_differences(img_H15, img_H16, img_H17, directory_path)

inputs_data = np.load("/notebooks/clean_data_H017/images_data_SQG.npy")
targets_data = np.load("/notebooks/clean_data_H017/flows_data_SQG.npy")

inputs_data = np.transpose(inputs_data, (0, 3, 1, 2))[:, np.newaxis, :, :, :]  # (1500, 2, 1, 256, 256)

# Reshape des cibles
targets_data = np.transpose(targets_data, (0, 3, 1, 2))  # (1500, 2, 256, 256)

print(f"Inputs shape : {inputs_data.shape}")
print(f"Targets shape : {targets_data.shape}")

# Ajouter la dimension de canal
#inputs_data = inputs_data[:, np.newaxis, :, :]

max_value = inputs_data.max()
#inputs_data = inputs_data / max_value

print("Valeur maximale :", max_value)
#print("Valeur maximale :", inputs_data.max())
# Vérifier la nouvelle forme
#print("Nouvelle forme des données :", inputs_data.shape)
#non_filtered = inputs_data
#inputs_data = apply_filters(inputs_data, filter_type="fft_lowpass")
#apply_filters(data, filter_type="gaussian")
#print(f"Shape after Filtering : {inputs_data.shape}")
#plot_filtering_comparison(np.squeeze(non_filtered, axis=1),np.squeeze(inputs_data, axis=1), directory_path)

total_steps = 1400
#num_features = 2  # u, v, w, p
spatial_size = (256, 256)  # Taille des dimensions spatiales (96x96x96)

# Calcul du nombre de séquences
num_sequences = total_steps - sequence_length

X = inputs_data[:total_steps]
Y = targets_data[:total_steps]
# Combinaison des données dans un seul array pour facilité
#X, Y = create_sequences(total_steps, sequence_length, inputs_data, targets_data, spatial_size)
#X, Y = create_sequences(total_steps, sequence_length, targets_data, targets_data, spatial_size)
# Split en ensembles de données
train_split = int(0.8*len(X))
val_split = int(0.8*len(X))
end_split = int(0.9*len(X))

X_train, Y_train = X[:train_split], Y[:train_split]
X_val, Y_val = X[val_split:end_split], Y[val_split:end_split]
X_test, Y_test = X[val_split:], Y[val_split:]

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

model =  UNetLSTM2D(num_classes=2, start_dim=start_dim, input_height=input_height, input_width=input_width).to(device)
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

        if val_loss <= 9e-4 and average_loss <= 9e-4:
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


min_timesteps = min(u_true.shape[0], u_pred.shape[0])

#E_u, E_v, E_p = zip(*[compute_metric_E(u_true[i], u_pred[i], v_true[i], v_pred[i], p_true[i], p_pred[i], mode="max") for i in range(min_timesteps)])

# Calcul de la métrique E(t) pour chaque variable (u, v, w, p)
#Emae_u = [compute_metric_MAEt(u_true[i, :, :], u_pred[i, :, :]) for i in range(min_timesteps)]
#Emae_v = [compute_metric_MAEt(v_true[i, :, :], v_pred[i, :, :]) for i in range(min_timesteps)]

end_time = time.time()
execution_time = end_time - start_time
# Convertir en heures, minutes, secondes
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = execution_time % 60

max_errors = []
x_min, x_max, y_min, y_max = 0, 6, 0, 2


for i in range(11, 16):
    print(f"Time Pred -- @ {i} step")
    error_u = u_pred - u_true
    error_v = v_pred - v_true
    #error_p = p_pred - p_true
    
    max_errors.append(np.abs(np.mean(error_u)))
    max_errors.append(np.abs(np.mean(error_v)))
    #max_errors.append(np.abs(np.mean(error_p)))
    plot_side_by_side(u_true[i], u_pred[i], error_u[i], i, "U", directory_picU, test_code, shading="flat")
    plot_side_by_side(v_true[i], v_pred[i], error_v[i], i, "V", directory_picV, test_code, shading="flat")
    #plot_side_by_side(p_true[i], p_pred[i], error_p[i], i, "P", directory_picP, test_code, x_min, x_max, y_min, y_max, shading="flat")
            

average_max_error = np.mean(max_errors)
FINAL_max_error = np.max(max_errors)

print(f"MOY des erreurs : {average_max_error}")
print(f"MAX des erreurs : {FINAL_max_error}")
#plot_error_metric(E_u, E_v, E_p, directory_path, test_code)
#plot_mae_metric(Emae_u, Emae_v, Emae_p, directory_path, test_code)

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

