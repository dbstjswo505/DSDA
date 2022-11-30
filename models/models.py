import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
# Dual-scale Doppler Attention
class DSDA(nn.Module):

    def __init__(self, input_dim, output_dim, dynamic_pooling=True):

        super(DSDA, self).__init__()

        doppler_pooling = np.zeros((4,), dtype=np.int32)
        time_pooling = np.zeros((4,), dtype=np.int32)
        # dynmiac poolling considers the features informations in the receptive fields
        # The feature is obtained from doppler signal
        if dynamic_pooling:
            doppler_pooling[:] = int((float(input_dim[2]) / 5) ** .25)
            res = ((float(input_dim[2]) / 5) ** .25) - doppler_pooling[0]
            for i in range(int(round(res * 4))):
                doppler_pooling[i] += 1
        else:
            doppler_pooling[:] = 2;
            c = 0
            while input_dim[2] < np.prod(doppler_pooling):
                doppler_pooling[-(c % 4) - 1] -= 1;
                c += 1

        if dynamic_pooling:
            time_pooling[:] = int((float(input_dim[1]) / 5) ** .25)
            res = ((float(input_dim[1]) / 5) ** .25) - time_pooling[0]
            for i in range(int(round(res * 4))):
                time_pooling[i] += 1
        else:
            time_pooling[:] = 2;
            c = 0
            while input_dim[1] < np.prod(time_pooling):
                time_pooling[-(c % 4) - 1] -= 1;
                c += 1

        self.encoder = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=input_dim[0], out_channels=8, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[0], doppler_pooling[0])),
            # Conv2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[1], doppler_pooling[1])),
            # Conv3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[2], doppler_pooling[2])),
            # Conv4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[3], doppler_pooling[3]))
        )  # --- Conv sequential ends ---

        self.jemb = nn.Sequential(
            nn.Linear(64 * 6 * 5, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )  # --- Linear sequential ends ---
        
        self.jemb2 = nn.Sequential(
            nn.Linear(64 * 18 * 5, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )  # --- Linear sequential ends ---
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * 64 * 18 * 5, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )  # --- Linear sequential ends ---


    def temporal_window_attention(self, x_twa, N=3):
        x1_twa = x_twa[0].view(-1,64*6*5)
        a1_twa = self.jemb(x1_twa).view(-1,1)

        x2_twa = x_twa[1].view(-1,64*6*5)
        a2_twa = self.jemb(x2_twa).view(-1,1)

        x3_twa = x_twa[2].view(-1,64*6*5)
        a3_twa = self.jemb(x3_twa).view(-1,1)

        a = torch.cat([a1_twa, a2_twa, a3_twa], dim=1)
        alpha = F.softmax(a, dim=1)
        
        x1 = x1_twa.view(-1, 1, 64*6*5)
        x2 = x2_twa.view(-1, 1, 64*6*5)
        x3 = x3_twa.view(-1, 1, 64*6*5)
        x = torch.cat([x1,x2,x3], dim=1)
        x = torch.einsum('bld,bn->bld',x,alpha)
        x = x.view(-1,64*18*5)

        return x
    
    def holistic_window_attention(self, x_hwa, x_sub, N=2):
        x1_twa = x_hwa.view(-1,64*18*5)
        a1_hwa = self.jemb2(x1_twa).view(-1,1)

        x2_twa = x_sub.view(-1,64*18*5)
        a2_sub = self.jemb2(x2_twa).view(-1,1)

        a = torch.cat([a1_hwa, a2_sub], dim=1)
        alpha = F.softmax(a, dim=1)
        
        x1 = x_hwa.view(-1, 1, 64*18*5)
        x2 = x_sub.view(-1, 1, 64*18*5)
        x = torch.cat([x1,x2], dim=1)
        x = torch.einsum('bld,bn->bd',x,alpha)
        x = x.view(-1,64*18*5)

        return x

    def window_encoder(self, x, N=3, T=50, S=50):
        # N is the number of windows
        x1 = self.encoder(x[:,:,:T,:])
        x2 = self.encoder(x[:,:,S:S+T,:])
        x3 = self.encoder(x[:,:,2*S:2*S+T,:])

        return [x1, x2, x3]

        

    def forward(self, x):
        x_twa = x
        pdb.set_trace()
        x_twa = self.window_encoder(x_twa, N=3, T=50, S=50)
        x_twa = self.temporal_window_attention(x_twa, N=3)

        x_hwa = x
        x_hwa = self.encoder(x_hwa)
        x_hwa = x_hwa.view(-1,64*18*5)
        x_sub = x_hwa - x_twa
        x_hwa = self.holistic_window_attention(x_hwa, x_sub, N=2)
        x = torch.cat([x_twa, x_hwa], dim=1)

        return self.classifier(x)



