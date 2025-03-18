import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.optim import Adam
from torch.autograd import Variable
from sklearn import preprocessing
import numpy as np


class AE(nn.Module):
    def __init__(self, n_input, n_enc_1, n_enc_2, n_z, n_dec_1, n_dec_2,
                 denoise=False, delta=2.0, pre_lr=0.01):
        super(AE, self).__init__()
        self.n_input = n_input
        self.n_enc_1 = n_enc_1
        self.n_enc_2 = n_enc_2
        self.n_z = n_z
        self.n_dec_1 = n_dec_1
        self.n_dec_2 = n_dec_2
        self.denoise = denoise
        self.delta = delta
        self.pre_lr = pre_lr

        self.enc_1 = Linear(self.n_input, self.n_enc_1, bias=True)
        self.enc_2 = Linear(self.n_enc_1, self.n_enc_2, bias=True)
        self._enc_pz = Linear(self.n_enc_2, self.n_z, bias=True)
        self.dec_1 = Linear(self.n_z, self.n_dec_1, bias=True)
        self.dec_2 = Linear(self.n_dec_1, self.n_dec_2, bias=True)
        self._dec_sigmoid = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), nn.Sigmoid())
        # self._dec_sigmoid = nn.Linear(self.n_dec_2, self.n_input)

    def encoder(self, x):
        if not self.denoise:
            enc_h1 = F.relu(self.enc_1(x))
            enc_h2 = F.relu(self.enc_2(enc_h1))
            pz = self._enc_pz(enc_h2)
        else:
            x = x + torch.randn_like(x) * self.delta
            enc_h1 = F.relu(self.enc_1(x))
            enc_h2 = F.relu(self.enc_2(enc_h1))
            pz = self._enc_pz(enc_h2)
        sigma, mu = torch.std_mean(pz, dim=0, keepdim=False)
        return pz, sigma, mu

    def decoder(self, z):
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        fz = self._dec_sigmoid(dec_h2)
        return fz

    def forward(self, x):
        pz, sigma, mu = self.encoder(x)
        fz = self.decoder(pz)
        # fz = F.relu(fz)
        return pz, fz, sigma, mu


def scale_minmax(data, axis=0):
    mi = np.expand_dims(data.min(axis=axis), axis=axis)
    ma = np.expand_dims(data.max(axis=axis), axis=axis)
    idx_0 = np.where((ma - mi) == 0)[0]
    s = np.zeros_like(ma)
    s[idx_0] = 1e-4
    data_scaled = (data - mi) / (ma - mi + s)
    return data_scaled


def calculate_loss_kl(pz, k, sigma, mu):
    loss_kl = 1 / 2 * (torch.sum((sigma + torch.pow(mu, 2)) - torch.log(sigma)) - k)
    return loss_kl


def get_exp_feature(data, scale_type=1):
    input_dim = data.shape[1]
    if scale_type == 0:
        data = scale_minmax(data, axis=0)
    elif scale_type == 1:
        data = scale_minmax(data, axis=1)
    elif scale_type == 2:
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
    # data = np.clip(a=data, a_min=1e-4, a_max=1-1e-4)
    vae_model = AE(n_input=input_dim, n_enc_1=64, n_enc_2=32, n_z=10,
                   n_dec_1=32, n_dec_2=64, denoise=False, delta=2.0,
                   pre_lr=0.01)

    vae_model.train()
    log_interval = 100
    epochs = 300
    optimizer = Adam(vae_model.parameters(), lr=vae_model.pre_lr)
    for epoch in range(1, epochs + 1):
        x_tensor = Variable(torch.Tensor(data))
        x_raw_tensor = torch.Tensor(data).clone()
        pz, fz, sigma, mu = vae_model(x_tensor)
        loss_reconst = torch.pow(torch.sum(torch.pow((x_raw_tensor - fz), 2)), 0.5) / input_dim
        loss_kl = calculate_loss_kl(pz, vae_model.n_z, sigma, mu)
        loss = loss_reconst + loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % log_interval == 0 or epoch == 1:
            print('\tVae_Epoch: {} ''\tvae_loss: {:.6f}'.format(epoch, loss.item()))
        if epoch == epochs:
            print('\n\tCompleted.')
    x_tensor = Variable(torch.Tensor(data))
    pz, fz, sigma, mu = vae_model(x_tensor)
    return pz
