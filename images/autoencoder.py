import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.rate = args.k/args.n
        self.SNR = round(10**(args.snr_db/10),4)
        self.in_dim = 2**args.k
        self.compressed_dim = args.n

        self.encoder = nn.Sequential(nn.Linear(self.in_dim,self.in_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.in_dim, self.compressed_dim))

        self.decoder = nn.Sequential(nn.Linear(self.compressed_dim,self.in_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.in_dim,self.in_dim))

    def encode_signal(self, x):
        return self.encoder(x).to(device)

    def decode_signal(self, x):
        return self.decoder(x)

    def AWGN(self, x):
        """ Adding Noise for testing step.
        """
        variance = 1 / (2 * self.rate * self.SNR)
        x = F.normalize(x,dim=1)      # 행 벡터에 대해서 normalize
        noise = torch.randn_like(x) * ((variance) ** 0.5)       # 논문에서 정의한 노이즈 가우시안 노이즈의 분산을 (2 * R*SNR)^(-1)로 정의했음
        x = x + noise
        return x

    def AWGN_var(self, x, SNR):
        """ Adding Noise for testing step.
        """
        variance = 1 / (2 * self.rate * SNR)
        x = F.normalize(x,dim=1)      # 행 벡터에 대해서 normalize
        noise = torch.randn_like(x) * ((variance) ** 0.5)       # 논문에서 정의한 노이즈 가우시안 노이즈의 분산을 (2 * R*SNR)^(-1)로 정의했음
        x = x + noise
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.AWGN(x)
        x = self.decoder(x)

        return x
