import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m=128, c_n=128, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv1d(in_channels, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.convB = nn.Conv1d(in_channels, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.convV = nn.Conv1d(in_channels, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.dropout = nn.Dropout(0.4)
        
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        
        self.global_pooling = nn.AdaptiveAvgPool1d(1) 
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        b, c, h = x.shape
        assert c == self.in_channels
        A = self.dropout(F.relu(self.convA(x)))
        B = self.dropout(F.relu(self.convB(x)))
        V = self.dropout(F.relu(self.convV(x)))
        
        tmpA = A.view(b, self.c_m, -1)  # b, c_m, h
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=-1)  
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=-1)  

        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1)) 
        tmpZ = global_descriptors.matmul(attention_vectors)  
        tmpZ = self.global_pooling(tmpZ)  
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)  
        tmpZ = tmpZ.view(b, -1) 

        return tmpZ

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_classes):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2_mean = nn.Linear(hidden_size, latent_size)  
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)  

        self.fc3 = nn.Linear(latent_size+633, hidden_size)   
        self.fc4 = nn.Linear(hidden_size, num_classes)  
        self.dropout = nn.Dropout(0.4)

        self.fc5 = nn.Linear(latent_size, latent_size)
        self.fc6 = nn.Linear(latent_size, latent_size)
        self.fc7 = nn.Linear(latent_size, latent_size)
        self.fc8 = nn.Linear(latent_size, latent_size)
        self.fc9 = nn.Linear(latent_size, latent_size)

        self.attention = DoubleAttention(in_channels=1)

    def zero_center(self, x):
        return x - x.mean(dim=0, keepdim=True)

    def diffusion(self, z, i):
        noise = torch.randn_like(z)  
        z_noisy = z + noise
        z_noisy = self.zero_center(z_noisy)
        z_noisy = self.attention(z_noisy)
        #print('z_noisy', z_noisy.shape) #z_noisy torch.Size([128, 256])
        z_denoised = self.dropout(F.relu(self.fc5(z_noisy)))  
        z_denoised = self.dropout(F.relu(self.fc6(z_denoised)))
        z_denoised = self.dropout(F.relu(self.fc7(z_denoised)))
        z_denoised = torch.cat((z_denoised, i), 1)
        return z_denoised
    
    def encode(self, x):
        h = self.dropout(F.relu(self.fc1(x)))
        mean = self.dropout(self.fc2_mean(h))
        logvar = self.dropout(self.fc2_logvar(h))
        return mean, logvar

    def reparameterize(self, mean, logvar): 
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std) 
        return mean + eps * std

    def decode(self, z):
        h = self.dropout(F.relu(self.fc3(z)))
        return self.fc4(h)

    def add_noise(self, z):
        noise_1 = torch.randn_like(z)
        noise_2 = torch.ones_like(z)
        noise_2 = self.fc8(noise_2)
        
        return noise_1*noise_2

    def forward(self, f, h, i):
        x = torch.cat((f, h), 1) 
        x = self.zero_center(x) 
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        #print('z', z.shape) #z torch.Size([64, 256])
        z_noise = self.add_noise(z)
        z = self.fc9(z) - z_noise
        z = F.relu(z)
        z = torch.cat((z, i), 1)
        z = self.decode(z)
        return z, mean, logvar

    def loss_function(self, output, label, mean, logvar):
        BCE = F.cross_entropy(output, label)  
        KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return BCE + KL