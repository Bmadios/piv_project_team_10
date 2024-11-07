import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, tri
import pandas as pd

def create_figure_and_axes(size=(18, 6)):
    fig, axs = plt.subplots(1, 3, figsize=size)  # Création de 3 subplots verticalement
    return fig, axs

def set_up_plot(ax):
    #rectangle = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, edgecolor='black', facecolor='none', linewidth=4)
    
    #ax.add_patch(rectangle)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    ax.set(xlabel='x ', ylabel=' y ')
    ax.axis('equal')

def plot_magnitude(data, title, ax, shading='flat'):
    set_up_plot(ax)  # Configure les limites et les formes géométriques pour chaque subplot
    # Afficher les données avec imshow sans définir de limites
    tpc = ax.imshow(data, cmap='jet', origin='lower', aspect='auto')

    # Ajustement de la colorbar
    cbar = plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    # Titre
    ax.set_title(title, fontweight="bold")
    
    #ax.set_title(title, fontweight="bold")


def plot_side_by_side(u_real, u_pred, error, t, field_name, folder, code, shading="flat"):
    fig, axs = create_figure_and_axes()
    plot_magnitude(u_real, f'{field_name} ground truth @ t={t} seconds', axs[0], shading)
    plot_magnitude(u_pred, f'{field_name} predicted with LSTM @ t={t} seconds', axs[1], shading)
    plot_magnitude(error, f'Absolute Error @ t={t} seconds', axs[2], shading)

    plt.tight_layout()
    plt.savefig(f"{folder}/Predicted_{field_name}_t{t}_{code}.png")
    plt.close(fig)

def plot_loss_rmse(train_losses, val_losses, train_rmses, val_rmses, loss_pic_path):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # First subplot for loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train and Validation Losses over Epochs')
    ax1.legend()
    ax1.grid(False)
    ax1.set_yscale('log')

    # Second subplot for RMSE
    ax2.plot(train_rmses, label='Train RMSE')
    ax2.plot(val_rmses, label='Validation RMSE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Train and Validation RMSE over Epochs')
    ax2.legend()
    ax2.grid(False)
    ax2.set_yscale('log')

    # Save the figure
    plt.savefig(loss_pic_path)
    plt.close(fig)  # Close the plot to free up memory


import numpy as np
import matplotlib.pyplot as plt
import os

def plot_error_metric(E_u, E_v, E_p, directory_path, test_code):
    # Calcul de l'erreur globale E(t)
    #E_global = np.sqrt(np.array(E_u)**2 + np.array(E_v)**2 + np.array(E_p)**2)
    
    # Trouver les valeurs maximales et leurs indices pour chaque série
    max_E_u, max_idx_E_u = max(E_u), np.argmax(E_u)
    max_E_v, max_idx_E_v = max(E_v), np.argmax(E_v)
    #max_E_w, max_idx_E_w = max(E_w), np.argmax(E_w)
    max_E_p, max_idx_E_p = max(E_p), np.argmax(E_p)
    #max_E_global, max_idx_E_global = max(E_global), np.argmax(E_global)
    #colors = {'u': 'blue', 'v': 'green', 'w': 'orange', 'p': 'purple', 'global': 'black'}
    colors = {'u': 'blue', 'v': 'green', 'p': 'red'}
    
    # Tracé de la métrique E(t) avec les valeurs maximales et l'erreur globale
    plt.figure(figsize=(10, 6))
    plt.plot(E_u, color=colors['u'])
    plt.plot(E_v, color=colors['v'])
    #plt.plot(E_w, label='E(t) for w')
    plt.plot(E_p, color=colors['p'])
    #plt.plot(E_global, linestyle='--', color='black', label='Global E(t)')
    
    # Marquer les points maximaux
    plt.plot(max_idx_E_u, max_E_u, 'o', label=f'Max E(t) u: {max_E_u:.2e}', color=colors['u'])
    plt.plot(max_idx_E_v, max_E_v, 'o', label=f'Max E(t) v: {max_E_v:.2e}', color=colors['v'])
    #plt.plot(max_idx_E_w, max_E_w, 'bo', label=f'Max E(t) w: {max_E_w:.2e}')
    plt.plot(max_idx_E_p, max_E_p, 'o', label=f'Max E(t) p: {max_E_p:.2e}', color=colors['p'])
    #plt.plot(max_idx_E_global, max_E_global, 'ko', label=f'Max Global E(t): {max_E_global:.2e}')
    
    plt.xlabel('Time Step')
    plt.ylabel('E(t)')
    plt.title('Error metric E(t) over time for u, v, p')
    #plt.legend()
    plt.legend(loc='upper right')
    #plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95), ncol=1, fontsize='small')
    plt.yscale('log')
    plt.ylim(1e-3, 7e-3) 
    plt.grid(True)
    
    # Sauvegarde du graphique
    plot_path = os.path.join(directory_path, f'E_metric_{test_code}.png')
    plt.savefig(plot_path)
    # plt.show()

def plot_mae_metric(Emae_u, Emae_v, Emae_p, directory_path, test_code):
    # Calcul de l'erreur globale MAE pour chaque pas de temps
    Emae_global = np.sqrt(np.array(Emae_u)**2 + np.array(Emae_v)**2 + np.array(Emae_p)**2)
    
    # Trouver les valeurs maximales et leurs indices pour chaque série
    max_Emae_u, max_idx_Emae_u = max(Emae_u), np.argmax(Emae_u)
    max_Emae_v, max_idx_Emae_v = max(Emae_v), np.argmax(Emae_v)
    #max_Emae_w, max_idx_Emae_w = max(Emae_w), np.argmax(Emae_w)
    max_Emae_p, max_idx_Emae_p = max(Emae_p), np.argmax(Emae_p)
    max_Emae_global, max_idx_Emae_global = max(Emae_global), np.argmax(Emae_global)
             
    colors = {'u': 'blue', 'v': 'green', 'p': 'red', 'global': 'black'}
    
    # Tracé de la métrique MAE avec les valeurs maximales et l'erreur globale
    plt.figure(figsize=(10, 6))
    plt.plot(Emae_u, color=colors['u'])
    plt.plot(Emae_v, color=colors['v'])
    #plt.plot(Emae_w, label='MAE for w')
    plt.plot(Emae_p, color=colors['p'])
    plt.plot(Emae_global, linestyle='--', color=colors['global'])
    
    # Marquer les points maximaux
    plt.plot(max_idx_Emae_u, max_Emae_u, 'o', label=f'Max MAE u: {max_Emae_u:.2e}', color=colors['u'])
    plt.plot(max_idx_Emae_v, max_Emae_v, 'o', label=f'Max MAE v: {max_Emae_v:.2e}', color=colors['v'])
    #plt.plot(max_idx_Emae_w, max_Emae_w, 'bo', label=f'Max MAE for w: {max_Emae_w:.2e}')
    plt.plot(max_idx_Emae_p, max_Emae_p, 'o', label=f'Max MAE p: {max_Emae_p:.2e}', color=colors['p'])
    plt.plot(max_idx_Emae_global, max_Emae_global, 'o', label=f'Max Global: {max_Emae_global:.2e}', color=colors['global'])
    
    plt.xlabel('Time Step')
    plt.ylabel('MAE(t)')
    plt.title('Error MAE(t) over time for u, v, p')
    #plt.legend()
    #plt.legend(loc='center', bbox_to_anchor=(1.05, 0.92), ncol=1, fontsize='small', frameon=False)
    plt.legend(loc='upper right')#, bbox_to_anchor=(1, 1.05), ncol=1, fontsize='small') #, bbox_to_anchor=(1, 0.95), ncol=1, fontsize='small')
    plt.yscale('log')
    plt.ylim(2e-2, 10e-2) 
    plt.grid(True)
    
    # Sauvegarde du graphique
    mae_plot_path = os.path.join(directory_path, f'E_MAE_overtime_{test_code}.png')
    plt.savefig(mae_plot_path)
    # plt.show()
