import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os

# LOAD DATA
arg_P = "H016"
PARENT_PATH = f"E:/Automne_2024/Projet_MLSP/Turbulent region/{arg_P}"
# Path to the image file
image_path = f'{PARENT_PATH}/MS_Images_{arg_P}/B00003.tif'
# Chemin vers le fichier de données .dat
data_path = f'E:/Automne_2024/Projet_MLSP/Turbulent region/H017/HDR_Data_H017/B00003.dat'
# Load the image using PIL
image = Image.open(image_path)
# Convert the image to a numpy array
image_array = np.array(image)
H1 = 206 
H2 = 326
W1 = 327
W2 = 447
# Crop the image using the grid indices
cropped_image = image_array#[H1:H2, W1:W2]

grid_x = np.arange(W1, W2)  
grid_y = np.arange(H1, H2)

# Verify the shape of the cropped image
print("Cropped image shape:", cropped_image.shape)

# Display the cropped image using matplotlib
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
plt.imshow(cropped_image, cmap='gray')
plt.colorbar()
#plt.title('Cropped 128x128 Image')
plt.title('Timestep 2')
plt.savefig(f"cropped_image_{arg_P}.png")

# Lire le fichier .dat, en ignorant les trois premières lignes de l'en-tête
data = pd.read_csv(data_path, delim_whitespace=True, skiprows=3, names=['X', 'Y', 'u', 'v', 'w', 'flag'])

valid_data = data[(data['X'].isin(grid_x)) & (data['Y'].isin(grid_y))]
print(valid_data)
#print(valid_data.shape)
grid_u = np.reshape(valid_data['u'].values, (len(grid_y), len(grid_x)))
grid_v = np.reshape(valid_data['v'].values, (len(grid_y), len(grid_x)))
grid_w = np.reshape(valid_data['w'].values, (len(grid_y), len(grid_x)))

#print(grid_u[100,100])
#print(grid_u[100,101])
#print(grid_u[100,102])
#print(data["flag"])

# Afficher le champ de vitesse u
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, grid_u, levels=100, cmap='jet')
plt.colorbar()
plt.title('Champ de Vitesse Horizontale (u)')
plt.xlabel('Position X (en pixels)')
plt.ylabel('Position Y (en pixels)')
#plt.show()
plt.savefig(f"Reference_U_solution_{arg_P}.png")
print(grid_u.shape)
# De même pour le champ de vitesse v
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, grid_v, levels=100, cmap='jet')
plt.colorbar()
plt.title('Champ de Vitesse Verticale (v)')
plt.xlabel('Position X (en pixels)')
plt.ylabel('Position Y (en pixels)')
#plt.show()
plt.savefig(f"Reference_V_solution_{arg_P}.png")

############# SAVE DATA FOR TRAINING MODEL  ##################

data_folder = f'{PARENT_PATH}/HDR_Data_{arg_P}'
file_list = [f for f in os.listdir(data_folder) if f.endswith('.dat') and f.startswith('B00')]
# Trier file_list en fonction du numéro dans les noms de fichiers
file_list.sort(key=lambda x: int(x[1:6]))

# Liste pour stocker les données de u, v, w dans un seul stack
all_data_stack = []

for file_name in file_list:
    print(file_name)
    file_path = os.path.join(data_folder, file_name)
    
    # Lire le fichier .dat, en ignorant les trois premières lignes de l'en-tête
    data = pd.read_csv(file_path, sep='\s+', skiprows=3, names=['X', 'Y', 'u', 'v', 'w', 'flag'])
    
    grid_x = np.arange(W1, W2)  
    grid_y = np.arange(H1, H2)
    valid_data = data[(data['X'].isin(grid_x)) & (data['Y'].isin(grid_y))]

    if valid_data.empty:
        print(f"Aucune donnée valide dans le fichier {file_name}")
        continue

    expected_size = len(grid_x) * len(grid_y)
    actual_size = len(valid_data['u'].values)
    
    if actual_size != expected_size:
        print(f"Le fichier {file_name} a une taille incompatible (données valides: {actual_size}, attendu: {expected_size}).")
        continue

    try:
        # Reshaper les données u, v, w en fonction des dimensions de la grille
        grid_u = np.reshape(valid_data['u'].values, (len(grid_y), len(grid_x)))
        grid_v = np.reshape(valid_data['v'].values, (len(grid_y), len(grid_x)))
        #grid_w = np.reshape(valid_data['w'].values, (len(grid_y), len(grid_x)))
        
        # Empiler u, v le long de la troisième dimension pour cette image
        data_stack = np.stack([grid_u, grid_v], axis=0)
        
        # Ajouter cette image empilée au stack global
        all_data_stack.append(data_stack)
        
    except ValueError as e:
        print(f"Erreur lors du reshape des données dans le fichier {file_name}: {e}")
        continue

# Convertir en tableau numpy et sauvegarder dans un fichier unique .npy
np.save(f'data_ground_truth_{arg_P}.npy', np.array(all_data_stack))

test_data = np.load(f'data_ground_truth_{arg_P}.npy')
print(test_data.shape)


image_folder = f'{PARENT_PATH}/MS_Images_{arg_P}'
image_list = [f for f in os.listdir(image_folder) if f.endswith('.tif') and f.startswith('B00') and f !="B00001.tif" and f!="B00002.tif"]
image_list.sort(key=lambda x: int(x[1:6]))
# Liste pour stocker toutes les images rognées
all_images_stack = []

for image_name in image_list:
    print(image_name)
    image_path = os.path.join(image_folder, image_name)
    
    # Charger l'image avec PIL
    image = Image.open(image_path)
    # Convertir l'image en tableau numpy
    image_array = np.array(image)
    # Rogner l'image
    cropped_image = image_array[H1:H2, W1:W2]
    # Ajouter l'image rognée au stack global
    all_images_stack.append(cropped_image)

# Convertir en tableau numpy et sauvegarder dans un fichier unique .npy
np.save(f'noisy_images_data_{arg_P}.npy', np.array(all_images_stack))

image_data = np.load(f'noisy_images_data_{arg_P}.npy')
print(image_data.shape)
