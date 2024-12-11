import torch.nn as nn
import torch
from torch.nn import Module, Linear, Tanh
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # device


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

def calculate_conv_output_size(input_size, kernel_size=3, padding=1, stride=1, dilation=1):
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def calculate_pool_output_size(input_size, kernel_size=2, stride=2):
    return (input_size - (kernel_size - 1) - 1) // stride + 1

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, kernel_size=3, padding=1, stride=1):
        super(ConvBlock2D, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConvBlock2D(nn.Module):
    def __init__(self, in_channels, middle_channels, skip_channels, out_channels, num_convs=2):
        super(UpConvBlock2D, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)
        self.conv_reduce = nn.Conv2d(middle_channels + skip_channels, out_channels, kernel_size=1)
        self.conv_block = ConvBlock2D(out_channels, out_channels, num_convs=num_convs)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_reduce(x)
        return self.conv_block(x)

class UNetLSTM2D(nn.Module):
    def __init__(self, input_dim, num_classes,input_height,input_width,num_convs=2,stride=1,start_dim=16):
        super(UNetLSTM2D, self).__init__()
        self.down1 = ConvBlock2D(input_dim, start_dim, num_convs=num_convs, stride=1)  # 4 channels for u, v, w, p
        self.down2 = ConvBlock2D(start_dim, start_dim*2, num_convs=num_convs, stride=stride)
        self.down3 = ConvBlock2D(start_dim*2, start_dim*4, num_convs=num_convs, stride=stride)
        self.down4 = ConvBlock2D(start_dim*4, start_dim*8, num_convs=num_convs, stride=stride)
        self.down5 = ConvBlock2D(start_dim*8, start_dim*16, num_convs=num_convs, stride=stride)
        self.pool = nn.MaxPool2d(2, 2)
        self.input_height = input_height
        self.input_width = input_width
        self.start_dim = start_dim
        self.input_dim = input_dim
        print(f"U_net Base: {start_dim}, Fin: {start_dim*16}")
        
        # Calcul des dimensions après chaque couche
        h, w = input_height, input_width
        h1, w1 = calculate_conv_output_size(h, stride=1), calculate_conv_output_size(w, stride=1)
        h1, w1 = calculate_pool_output_size(h1), calculate_pool_output_size(w1)
        
        h2, w2 = calculate_conv_output_size(h1, stride=stride), calculate_conv_output_size(w1, stride=stride)
        h2, w2 = calculate_pool_output_size(h2), calculate_pool_output_size(w2)
        
        h3, w3 = calculate_conv_output_size(h2, stride=stride), calculate_conv_output_size(w2, stride=stride)
        h3, w3 = calculate_pool_output_size(h3), calculate_pool_output_size(w3)
        
        h4, w4 = calculate_conv_output_size(h3, stride=stride), calculate_conv_output_size(w3, stride=stride)
        h4, w4 = calculate_pool_output_size(h4), calculate_pool_output_size(w4)
        
        h5, w5 = calculate_conv_output_size(h4, stride=stride), calculate_conv_output_size(w4, stride=stride)
        h5, w5 = calculate_pool_output_size(h5), calculate_pool_output_size(w5)
        
        # Assurez-vous que les dimensions ne deviennent pas nulles
        if h4 == 0 or w4 == 0:
            raise ValueError("Les dimensions après les convolutions et les poolings sont nulles. Veuillez ajuster les paramètres du réseau (stride, padding, etc.) pour éviter cela.")
        
        #print(h4, w4)
        print(h5, w5)
        print(h4, w4)
        # Taille aplatie pour l'entrée du LSTM
        #lstm_input_size = start_dim*8*h3*w3  # 128 canaux de down4
        lstm_input_size = start_dim*16*h4*w4  # 128 canaux de down4
        #lstm_input_size = start_dim*8
        print(f"LSTM Input :{lstm_input_size}")
        
        #self.lstm = nn.LSTM(lstm_input_size, start_dim*16, num_layers=2, batch_first=True, dropout=0.5)
        self.lstm = nn.LSTM(lstm_input_size, start_dim*16, num_layers=1, batch_first=True)
        self.up4 = UpConvBlock2D(start_dim*16, start_dim*8, start_dim*8, start_dim*8, num_convs=num_convs)
        self.up3 = UpConvBlock2D(start_dim*8, start_dim*4, start_dim*4, start_dim*4, num_convs=num_convs)
        self.up2 = UpConvBlock2D(start_dim*4, start_dim*2, start_dim*2, start_dim*2, num_convs=num_convs)
        self.up1 = UpConvBlock2D(start_dim*2, start_dim, start_dim, start_dim, num_convs=num_convs)
        self.final_conv = nn.Conv2d(start_dim, num_classes, kernel_size=1)


    def forward(self, x):
        b, t, c, h, w = x.size()
        #print(x.shape)
        x = x.view(b * t, c, h, w)

        # Passer les données dans les blocs U-Net
        x1 = self.down1(x)
        p1 = self.pool(x1)
        x2 = self.down2(p1)
        p2 = self.pool(x2)
        x3 = self.down3(p2)
        p3 = self.pool(x3)
        x4 = self.down4(p3)
        
        p4 = self.pool(x4)
        x5 = self.down5(p4)

        # Redimensionner pour correspondre à l'entrée du LSTM
        _, c, h5, w5 = x5.size()
        x5 = x5.view(b, t, -1)
        lstm_out, _ = self.lstm(x5)
        
        # Utiliser la dernière sortie du LSTM pour remonter
        lstm_out = lstm_out[:, -1, :]
        #print(lstm_out.shape)
        lstm_out = lstm_out.view(b, self.start_dim*16, 1, 1)

        # Décompression à partir de up4 avec les skip connections
        x4 = x4.view(b, t, -1, x4.size(2), x4.size(3)).mean(dim=1)
        x = self.up4(lstm_out, x4)

        x3 = x3.view(b, t, -1, x3.size(2), x3.size(3)).mean(dim=1)
        x = self.up3(x, x3)

        x2 = x2.view(b, t, -1, x2.size(2), x2.size(3)).mean(dim=1)
        x = self.up2(x, x2)

        x1 = x1.view(b, t, -1, x1.size(2), x1.size(3)).mean(dim=1)
        x = self.up1(x, x1)

        # Appliquer la dernière convolution
        x = self.final_conv(x)

        # Interpolation à la taille d'entrée d'origine
        x = F.interpolate(x, size=(self.input_height, self.input_width), mode='bilinear', align_corners=False)

        return x