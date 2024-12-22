import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import optuna  # Pour l'optimisation des hyperparamètres
from models_2d import UNetLSTM2D

from utilis import *
from sp_functions import *
from plotting import *

###############################################################
# Configuration globale
###############################################################
start_time = time.time()

# Paramètres fixes
input_height = 256
input_width = 256
num_epochs = 600
batch_size = 32
step_size = 200

mode = "Train"
test_code = "10_Dec_TC_02_Method_00_Optuna"
USE_SEQ_IMG = False  # False = use Pair Images

if USE_SEQ_IMG:
    input_dim = 1
else:
    input_dim = 2

sequence_length = 1

directory_path = os.path.join('Final_tests', test_code)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

###############################################################
# Chargement des données
###############################################################
targets_data_H17 = np.load("/notebooks/clean_data_H017/data_ground_truth_H017.npy")
inputs_data_H17 = np.load("/notebooks/clean_data_H017/noisy_images_data_H017.npy")

targets_data_H16 = np.load("/notebooks/clean_data_H017/data_ground_truth_H016.npy")
inputs_data_H16 = np.load("/notebooks/clean_data_H017/noisy_images_data_H016.npy")

targets_data_H15 = np.load("/notebooks/clean_data_H017/data_ground_truth_H015.npy")
inputs_data_H15 = np.load("/notebooks/clean_data_H017/noisy_images_data_H015.npy")

inputs_data_H15, targets_data_H15 = create_consecutive_pairs_v2(inputs_data_H15, targets_data_H15)
inputs_data_H16, targets_data_H16 = create_consecutive_pairs_v2(inputs_data_H16, targets_data_H16)
inputs_data_H17, targets_data_H17 = create_consecutive_pairs_v2(inputs_data_H17, targets_data_H17)

inputs_data_temp = np.concatenate((inputs_data_H15, inputs_data_H16), axis=0)
targets_data_temp = np.concatenate((targets_data_H15, targets_data_H16), axis=0)
inputs_data = np.concatenate((inputs_data_temp, inputs_data_H17), axis=0)
targets_data = np.concatenate((targets_data_temp, targets_data_H17), axis=0)

max_value = inputs_data.max()
inputs_data = inputs_data / max_value


def permute_dimensions(tensor):
    """Permute dimensions if single sequence images are used."""
    return tensor.transpose(0, 2, 1, 3, 4)

X = inputs_data
Y = targets_data

if input_dim == 1:
    X = permute_dimensions(X)

total_steps = X.shape[0]

# Splits
train_split = int(0.8 * len(X))
val_split = int(0.85 * len(X))  # 10% validation
X_train, Y_train = X[:train_split], Y[:train_split]
X_val, Y_val = X[train_split:val_split], Y[train_split:val_split]
X_test, Y_test = X[val_split:], Y[val_split:]

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = CustomDataset(X_test_tensor, Y_test_tensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################################
# Fonction d'objectif pour Optuna
###############################################################
def objective(trial):
    # Echantillonnage des hyperparamètres
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    start_dim = trial.suggest_int("start_dim", 4, 32, step=4)
    batch_size = trial.suggest_int("batch_size", 4, 32, step=4)

    # Modèle et optimiseur
    model = UNetLSTM2D(input_dim=input_dim, num_classes=2, start_dim=start_dim, 
                       input_height=input_height, input_width=input_width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Entraînement (cycles réduits pour Optuna, ex: 50 époques max)
    max_optuna_epochs = 100
    best_val_aee = float('inf')

    for epoch in range(max_optuna_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluation sur validation
        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for data_val, target_val in val_loader:
                data_val = data_val.to(device)
                pred_val = model(data_val).cpu().numpy()
                val_predictions.append(pred_val)
                val_targets.append(target_val.numpy())

        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        # Calcul AEE sur val
        u_pred = val_predictions[:, 0]
        v_pred = val_predictions[:, 1]
        u_true = val_targets[:, 0]
        v_true = val_targets[:, 1]
        val_aee = compute_aee(u_pred, u_true, v_pred, v_true)

        # Enregistrement du meilleur AEE
        if val_aee < best_val_aee:
            best_val_aee = val_aee

        # Enregistrement des résultats de chaque essai
        with open(os.path.join(directory_path, "Optuna_trials.txt"), "a") as f:
            f.write(f"Trial {trial.number}, Epoch {epoch}, LR: {learning_rate}, start_dim: {start_dim}, Val_AEE: {val_aee}\n")

        # Possibilité d'utiliser un early stopping rudimentaire
        trial.report(val_aee, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_aee

###############################################################
# Lancement de l'étude Optuna
###############################################################
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # Ajuster le nombre d'essais

best_trial = study.best_trial
best_learning_rate = best_trial.params["learning_rate"]
best_start_dim = best_trial.params["start_dim"]
best_batch_size = best_trial.params["batch_size"]

# Enregistrer les meilleurs hyperparamètres
with open(os.path.join(directory_path, "Best_Optuna_Params.txt"), "w") as f:
    f.write(f"Best Trial: {best_trial.number}\n")
    f.write(f"Best Learning Rate: {best_learning_rate}\n")
    f.write(f"Best start_dim: {best_start_dim}\n")
    f.write(f"Best batch_size : {best_batch_size}\n")
    f.write(f"Best Val AEE: {best_trial.value}\n")

###############################################################
# Réentraînement du modèle final avec les meilleurs hyperparams
###############################################################
model = UNetLSTM2D(input_dim=input_dim, num_classes=2, start_dim=best_start_dim, 
                   input_height=input_height, input_width=input_width).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
criterion = torch.nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if mode == "Train":
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

    # Sauvegarde du modèle final
    model_file_path = os.path.join(directory_path, f'best_model_UnetLSTM2D_{test_code}.pth')
    torch.save(model.state_dict(), model_file_path)

model.eval()
test_predictions = predict(model, test_loader, device)

# Création de répertoires pour les images
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


min_timesteps = min(u_true.shape[0], u_pred.shape[0])

E_u, E_v, _ = zip(*[compute_metric_E(u_true[i], u_pred[i], v_true[i], v_pred[i], v_true[i], v_pred[i], mode="min")
                    for i in range(min_timesteps)])
Emae_u = [compute_metric_MAEt(u_true[i], u_pred[i]) for i in range(min_timesteps)]
Emae_v = [compute_metric_MAEt(v_true[i], v_pred[i]) for i in range(min_timesteps)]

end_time = time.time()
execution_time = end_time - start_time
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = execution_time % 60

select_times = [10, 20, 30, 40, 49, 60, 70, 80, 100]
error_u = u_pred - u_true
error_v = v_pred - v_true

for i in select_times:
    if i < len(u_true):
        plot_side_by_side(u_true[i], u_pred[i], error_u[i], i+1, "U", directory_picU, test_code, shading="flat")
        plot_side_by_side(v_true[i], v_pred[i], error_v[i], i+1, "V", directory_picV, test_code, shading="flat")

rmse = compute_rmse(u_pred, u_true, v_pred, v_true)
aee = compute_aee(u_pred, u_true, v_pred, v_true)

plot_error_metric(E_u, E_v, directory_path, test_code)
plot_mae_metric(Emae_u, Emae_v, directory_path, test_code)

# Sauvegarde des résultats finaux
results_path = os.path.join(directory_path, f'Results_{test_code}.txt')
with open(results_path, "w") as f:
    f.write(f"**** TEST CONFIG ({test_code}) ****\n")
    f.write(f"Sequence Length: {sequence_length}\n")
    f.write(f"Batch size: {batch_size}\n\n")
    f.write(f"**** BEST HYPERPARAMS FROM OPTUNA ****\n")
    f.write(f"Learning Rate: {best_learning_rate}\n")
    f.write(f"start_dim: {best_start_dim}\n\n")
    f.write(f"**** TRAINING CONFIG ****\n")
    f.write(f"Epochs: {num_epochs}\n\n")
    f.write(f"**** RESULTS ****\n")
    f.write(f"Execution time: {hours}h:{minutes}m:{seconds:.2f}s\n")
    f.write(f"Final RMSE: {rmse}\n")
    f.write(f"Final AEE: {aee}\n")
