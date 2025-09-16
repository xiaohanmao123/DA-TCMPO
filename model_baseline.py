import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_classes):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  
        )
        self.fc1 = nn.Linear(2169, 633)  

    def forward(self, f, h, i):
        x = torch.cat((f, h), 1) 
        #print('x', x.shape, 'i', i.shape) #x torch.Size([64, 1536]) i torch.Size([64, 633])
        x = self.encoder(x.unsqueeze(1))
        #print('x', x.shape) #x torch.Size([64, 64, 384])
        x = self.decoder(x)
        #print('x', x.shape) #x torch.Size([64, 1, 1536])
        x = torch.cat((x.squeeze(1), i), 1)
        x = self.fc1(x)
        
        return x

    def loss_function(self, output, label):
        #print('output', output.shape, 'label', label.shape, 'mean', mean.shape, 'logvar', logvar.shape) #output torch.Size([64, 633]) label torch.Size([64])
        BCE = F.cross_entropy(output, label)  
        return BCE