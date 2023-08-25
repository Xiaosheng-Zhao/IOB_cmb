from torch import nn

### models
### Fully connected
class Encoder_cmb(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        self.encoder_nn = nn.Sequential(
            nn.Linear(encoded_space_dim, 96),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.encoder_nn(x)
        return x
    
class maskin_cmb(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.maskin_nn = nn.Linear(96,6)
        
    def forward(self,x):
        x = self.maskin_nn(x)
        return x
    
    
class Decoder_cmb(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        self.decoder_nn = nn.Sequential(
            #nn.Linear(encoded_space_dim, 256),
            #nn.ReLU(True),
            nn.Linear(6, 512),
            nn.ReLU(True),
            nn.Linear(512, 2499)
        )
    def forward(self, x):
        x = self.decoder_nn(x)
        #x = torch.sigmoid(x)
        return x
