import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .VAE_base import Model
from .utils import  NonLinear, GatedDense
from .mcevae_config import MCEVAEConfig

from ...data.datasets import BaseDataset
from ..ae import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder



class MCEVAE(Model):
    """
    Disentangled :math:`\MCE`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 10.
        C (float): The value of the KL divergence term of the ELBO we wish to approach, measured in
            nats. Default: 50.
        warmup_epoch (int): The number of epochs during which the KL divergence objective will
            increase from 0 to C (should be smaller or equal to nb_epochs). Default: 100
        epoch (int): The current epoch. Default: 0
    """
    def __init__(self,model_config: MCEVAEConfig, 
                in_size=28*28,
                aug_dim=16*7*7,
                latent_z_c=0,
                latent_z_var=5,
                mode='SO2', 
                invariance_decoder='gated', 
                rec_loss='mse', 
                div='KL',
                in_dim=1, 
                out_dim=1, 
                hidden_z_c=300,
                hidden_z_var=300,
                hidden_tau=32, 
                activation=nn.Sigmoid,
                training_mode = 'supervised',
                device = 'cpu',
                tag = 'default'):
        super(MCEVAE, self).__init__()
        self.mode = model_config.mode
        self.invariance_decoder = model_config.invariance_decoder
        self.rec_loss = model_config.rec_loss
        self.div_mode = model_config.div
        self.hidden_z_c = model_config.hidden_z_c
        self.hidden_z_var = model_config.hidden_z_var
        self.hidden_tau = model_config.hidden_tau
        self.latent_z_c = model_config.latent_z_c
        self.latent_z_var = model_config.latent_z_var
        self.in_dim = model_config.in_dim
        self.out_dim = model_config.out_dim
        self.aug_dim = model_config.aug_dim
        self.in_size = model_config.in_size
        self.device= model_config.device
        self.training_mode = model_config.training_mode
        self.tag = model_config.tag

        print('in_size: {}, latent_z_c: {}, latent_z_var:{}, mode: {}, sem_dec: {}, rec_loss: {}, div: {}'.format(in_size, latent_z_c, latent_z_var, mode, invariance_decoder, rec_loss, div))
        
        # transformation type
        if mode == 'SO2':
            tau_size = 1
            bias = torch.tensor([0], dtype=torch.float)
        elif mode == 'SE2':
            tau_size = 3
            bias = torch.tensor([0, 0, 0], dtype=torch.float)
        elif mode == 'SIM2':
            tau_size = 4
            bias = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        elif mode == 'SE3':
            tau_size = 6
            bias = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float)
        elif mode == 'NONE':
            tau_size = 0
            bias = torch.tensor([], dtype=torch.float)
        else:
            raise NotImplementedError
        self.tau_size = tau_size
        
        # augmented encoder
        self.aug_enc = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2, bias=True)
        )
        
        # transformation extractor
        if tau_size > 0:
            self.tau_mean = nn.Sequential(
                nn.Linear(self.aug_dim, hidden_tau),
                nn.ReLU(True),
                nn.Linear(hidden_tau, tau_size)
            )
        
            self.tau_logvar = nn.Sequential(
                nn.Linear(self.aug_dim, hidden_tau),
                nn.ReLU(True),
                nn.Linear(hidden_tau, tau_size)
            )
        
            self.tau_mean[2].weight.data.zero_()
            self.tau_mean[2].bias.data.copy_(bias)
            self.tau_logvar[2].weight.data.zero_()
            self.tau_logvar[2].bias.data.copy_(bias)

        # Variational latent space extractor
        if self.latent_z_var > 0:
            self.q_z_var_mean = nn.Sequential(
                nn.Linear(self.aug_dim, hidden_z_var),
                nn.Sigmoid(),
                nn.Linear(hidden_z_var, hidden_z_var),
                nn.Sigmoid(),
                nn.Linear(hidden_z_var, latent_z_var)
            )
        
            self.q_z_var_logvar = nn.Sequential(
                nn.Linear(self.aug_dim, hidden_z_var),
                nn.Sigmoid(),
                nn.Linear(hidden_z_var, hidden_z_var),
                nn.Sigmoid(),
                nn.Linear(hidden_z_var, latent_z_var)
            )
        
        # semantic/shape extractor 2 = entangled latent space extractor
        if self.latent_z_c > 0:
            self.q_z_c_mean = nn.Sequential(
                nn.Linear(self.aug_dim, hidden_z_c),
                nn.Sigmoid(),
                nn.Linear(hidden_z_c, hidden_z_c),
                nn.Sigmoid(),
                nn.Linear(hidden_z_c, latent_z_c)
            )
        
            self.q_z_c_logvar = nn.Sequential(
                nn.Linear(self.aug_dim, hidden_z_c),
                nn.Sigmoid(),
                nn.Linear(hidden_z_c, hidden_z_c),
                nn.Sigmoid(),
                nn.Linear(hidden_z_c, latent_z_c)
            )
        
        # invariance decoder
        if invariance_decoder == 'linear':
            self.p_x_layer = nn.Sequential(
                nn.Linear(latent_z_c + latent_z_var, hidden_z_c),
                activation(),
                nn.Linear(hidden_z_c, hidden_z_c),
                activation(),
                nn.Linear(hidden_z_c, hidden_z_c),
                activation(),
                nn.Linear(hidden_z_c, np.prod(in_size))
            )
        elif invariance_decoder == 'gated':
            self.p_x_layer = nn.Sequential(
                GatedDense(latent_z_c + latent_z_var, 300),
                GatedDense(300, 300),
                NonLinear(300, np.prod(in_size), activation=activation())
            )
        elif invariance_decoder == 'CNN':
            self.sem_dec_fc = nn.Linear(latent_z_c + latent_z_var, self.aug_dim)
            self.p_x_layer = nn.Sequential(
                nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1, padding=2, output_padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1,  output_padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.ConvTranspose2d(16, out_dim, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=True)
            )

    def loss_function(model, x, x_init, beta=1., n_sampel=4):

        # print('x is ', x.size())
        x_hat, z_var_q, z_var_q_mu, z_var_q_logvar, \
        z_c_q, z_c_q_mu, z_c_q_logvar, z_c_q_L, tau_q, tau_q_mu, tau_q_logvar, x_rec, M = model(x)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        x = x.view(-1, model.in_size).to(device)
        x_hat = x_hat.view(-1, model.in_size)
        x_rec = x_rec.view(-1, model.in_size)
            
        if model.rec_loss == 'mse':
            RE = torch.sum((x - x_hat)**2)
            if model.tau_size > 0 and model.training_mode == 'supervised':
                RE_INV = torch.sum((x_rec - x_init)**2)
            elif model.tau_size > 0 and model.training_mode == 'unsupervised':
                RE_INV = torch.FloatTensor([0.]).to(device)
                for jj in range(25):
                    with torch.no_grad():
                        x_arb = model.get_x_ref(x.view(-1,1,int(np.sqrt(model.in_size)),int(np.sqrt(model.in_size))), tau_q)
                        z_aug_arb = model.aug_encoder(x_arb)
                        z_c_q_mu_arb, z_c_q_logvar_arb, _ = model.q_z_c(z_aug_arb)
                        z_c_q_arb = model.reparameterize(z_c_q_mu_arb, z_c_q_logvar_arb).to(device)
                        z_var_q_mu_arb, z_var_q_logvar_arb = model.q_z_var(z_aug_arb)
                        z_var_q_arb = model.reparameterize(z_var_q_mu_arb, z_var_q_logvar_arb).to(device)
                        x_init, _ = model.reconstruct(z_var_q_arb, z_c_q_arb)
                        x_init = x_init.view(-1, model.in_size).to(device)
                        x_init = torch.clamp(x_init, 1.e-5, 1-1.e-5)
                    RE_INV = RE_INV + torch.sum((z_var_q_arb - z_var_q)**2)
                    RE_INV = RE_INV + torch.sum((z_c_q_arb - z_c_q)**2) 
                    RE_INV = RE_INV + torch.sum((x_rec - x_init)**2)
                RE_INV = RE_INV/25.0
            else:
                RE_INV = torch.FloatTensor([0.]).to(device)
        elif model.rec_loss == 'bce':
            x_hat = torch.clamp(x_hat, 1.e-5, 1-1.e-5)
            x = torch.clamp(x, 1.e-5, 1-1.e-5)
            x_init = torch.clamp(x_init, 1.e-5, 1-1.e-5)
            x_rec = torch.clamp(x_rec, 1.e-5, 1-1.e-5)
            RE = -torch.sum((x*torch.log(x_hat) + (1-x)*torch.log(1-x_hat)))
            if model.tau_size > 0 and model.training_mode == 'supervised':
                x_init = x_init.view(-1, model.in_size).to(device)
                RE_INV = -torch.sum((x_init*torch.log(x_rec) + (1-x_init)*torch.log(1-x_rec)))
            elif model.tau_size > 0 and model.training_mode == 'unsupervised':
                RE_INV = torch.FloatTensor([0.]).to(device)
                for jj in range(25):
                    with torch.no_grad():
                        x_arb = model.get_x_ref(x.view(-1,1,int(np.sqrt(model.in_size)),int(np.sqrt(model.in_size))), tau_q)
                        z_aug_arb = model.aug_encoder(x_arb)
                        z_c_q_mu_arb, z_c_q_logvar_arb, _ = model.q_z_c(z_aug_arb)
                        z_c_q_arb = model.reparameterize(z_c_q_mu_arb, z_c_q_logvar_arb).to(device)
                        z_var_q_mu_arb, z_var_q_logvar_arb = model.q_z_var(z_aug_arb)
                        z_var_q_arb = model.reparameterize(z_var_q_mu_arb, z_var_q_logvar_arb).to(device)
                        x_init, _ = model.reconstruct(z_var_q_arb, z_c_q_arb)
                        x_init = x_init.view(-1, model.in_size).to(device)
                        x_init = torch.clamp(x_init, 1.e-5, 1-1.e-5)
                    RE_INV = RE_INV + torch.sum((z_var_q_arb - z_var_q)**2)
                    RE_INV = RE_INV + torch.sum((z_c_q_arb - z_c_q)**2) 
                    RE_INV = RE_INV - torch.sum((x_init*torch.log(x_rec) + (1-x_init)*torch.log(1-x_rec)))
                RE_INV = RE_INV/25.0
            else:
                RE_INV = torch.FloatTensor([0.]).to(device)
        else:
            raise NotImplementedError

        if z_var_q.size()[0] == 0:
            log_q_z_var, log_p_z_var = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
        else:
            log_q_z_var = -torch.sum(0.5*(1 + z_var_q_logvar))
            log_p_z_var = -torch.sum(0.5*(z_var_q**2 )) 
            
        if tau_q.size()[0] == 0:
            log_q_tau, log_p_tau = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
        else:
            log_q_tau = -torch.sum(0.5*(1 + tau_q_logvar))
            log_p_tau = -torch.sum(0.5*(tau_q**2 ))
        if z_c_q.size()[0] == 0:
            log_q_z_c, log_p_z_c = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
        else:
            log_q_z_c = -torch.sum(0.5*(1 + z_c_q_logvar/model.latent_z_c + \
                                        (model.latent_z_c -1)*z_c_q**2/model.latent_z_c))
            log_p_z_c = -torch.sum(0.5*(z_c_q**2 )) + torch.sum(z_c_q)/model.latent_z_c

        likelihood = - (RE + RE_INV)/x.shape[0]
        divergence_c = (log_q_z_c - log_p_z_c)/x.shape[0]
        divergence_var_tau = (log_q_z_var - log_p_z_var)/x.shape[0]  + (log_q_tau - log_p_tau)/x.shape[0]


        loss = - likelihood + beta * divergence_var_tau + divergence_c

        return (
            loss,
            RE/x.shape[0],
            divergence_var_tau,
            divergence_c
        )



    def aug_encoder(self, x):
        x = self.aug_enc(x)
        z_aug = x.view(-1, self.aug_dim)
        return z_aug

    def q_z_var(self, z_aug):
        if self.latent_z_var == 0:
            return torch.FloatTensor([]), torch.FloatTensor([])
        z_var_q_mu = self.q_z_var_mean(z_aug)
        z_var_q_logvar = self.q_z_var_logvar(z_aug)
        return z_var_q_mu, z_var_q_logvar
    
    
    def q_z_c(self, z_aug):
        if self.latent_z_c == 0:
            return torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([]) 
        z_c_q_mu = self.q_z_c_mean(z_aug)
        z_c_q_logvar = self.q_z_c_logvar(z_aug)
        return z_c_q_mu, z_c_q_logvar, torch.FloatTensor([])

    
    def q_tau(self, z_aug):
        if self.tau_size == 0:
            return torch.FloatTensor([]), torch.FloatTensor([])
        tau_q_mu = self.tau_mean(z_aug)
        tau_q_logvar = self.tau_logvar(z_aug)
        return tau_q_mu, tau_q_logvar


    def get_M(self, tau):
        if self.tau_size == 0:
            return 1., 0.
        params = torch.FloatTensor(tau.size()).fill_(0)
        if self.mode == 'SO2':
            M = torch.FloatTensor(tau.size()[0], 2, 3).fill_(0)
            theta = tau.squeeze()
            M[:, 0, 0] = torch.cos(theta)
            M[:, 0, 1] = -1 * torch.sin(theta)
            M[:, 1, 0] = torch.sin(theta)
            M[:, 1, 1] = torch.cos(theta)
            params = theta
        elif self.mode == 'SE2':
            M = torch.FloatTensor(tau.size()[0], 2, 3).fill_(0)
            theta = tau[:, 0] + 1.e-20
            u_1 = tau[:, 1]
            u_2 = tau[:, 2]
            M[:, 0, 0] = torch.cos(theta)
            M[:, 0, 1] = -1 * torch.sin(theta)
            M[:, 1, 0] = torch.sin(theta)
            M[:, 1, 1] = torch.cos(theta)
            M[:, 0, 2] = u_1 / theta * torch.sin(theta) - u_2 / theta * (1 - torch.cos(theta))
            M[:, 1, 2] = u_1 / theta * (1 - torch.cos(theta)) + u_2 / theta * torch.sin(theta)
            params[:, 0] = tau[:, 0]
            params[:, 1:] = M[:, :, 2]
        elif self.mode == 'SIM2':
            M = torch.FloatTensor(tau.size()[0], 2, 3).fill_(0)
            theta = tau[:, 0] + 1.e-20
            u_1 = tau[:, 1]
            u_2 = tau[:, 2]
            scale = tau[:, 3].reshape(-1,1,1).cpu()
            M[:, 0, 0] = torch.cos(theta)
            M[:, 0, 1] = -1 * torch.sin(theta)
            M[:, 1, 0] = torch.sin(theta)
            M[:, 1, 1] = torch.cos(theta)
            M[:, 0, 2] = u_1 / theta * torch.sin(theta) - u_2 / theta * (1 - torch.cos(theta))
            M[:, 1, 2] = u_1 / theta * (1 - torch.cos(theta)) + u_2 / theta * torch.sin(theta)
            M = M*scale
            params[:, 0] = tau[:, 0]
            params[:, 1:3] = M[:, :, 2]
        return M, params

    def reconstruct(self, z_var, z_c):
        # concatenating the vectors of z_var and z_c
        z = torch.cat((z_var, z_c), dim=1)
        if self.invariance_decoder == 'CNN':
            x = self.sem_dec_fc(z)
            x = x.view(-1, 16, 7, 7)
            x_mean = self.p_x_layer(x)
            x_mean = torch.sigmoid(x_mean)
        else:
            x_mean = self.p_x_layer(z)
        x_min = 1. / 512.
        x_max = 1 - x_min
        x_rec = torch.clamp(x_mean, min=x_min, max=x_max)
        return x_rec, 0.

    def transform(self, x, M, direction='forward', padding_mode='zeros'):
        if self.tau_size == 0:
            return x
        if direction == 'reverse':
            M_rev = torch.FloatTensor(M.size()).fill_(0)
            R_rev = torch.inverse(M[:, :, :2].squeeze())
            t = M[:, :, 2:]
            t_rev = torch.matmul(R_rev, t).squeeze()
            M_rev[:, :, :2] = R_rev
            M_rev[:, :, 2] = -1 * t_rev
            M = M_rev
        elif direction != 'forward':
            raise NotImplementedError
        grid = F.affine_grid(M, x.size(),align_corners=False).to(self.device)
        x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
        return x
    
    def forward(self,model, x):
        z_aug = self.aug_encoder(x)
        z_var_q_mu, z_var_q_logvar = self.q_z_var(z_aug)
        z_var_q = self.reparameterize(z_var_q_mu, z_var_q_logvar).to(self.device)
        z_c_q_mu, z_c_q_logvar, z_c_q_L = self.q_z_c(z_aug)
        z_c_q = self.reparameterize(z_c_q_mu, z_c_q_logvar, z_c_q_L).to(self.device)
        tau_q_mu, tau_q_logvar = self.q_tau(z_aug)        
        tau_q = self.reparameterize(tau_q_mu, tau_q_logvar).to(self.device)
        M, params = self.get_M(tau_q)
        x_rec, _ = self.reconstruct(z_var_q, z_c_q) #nn.Softmax().forward(z_c_q))
        x_rec = x_rec.view(-1, 1, int(np.sqrt(self.in_size)), int(np.sqrt(self.in_size)))
        x_hat = self.transform(x_rec, M, direction='forward')

        loss, reco_loss, divergence_var_tau, divergence_c = self.loss_function(model, x, x_hat, beta = 1)

        output = ModelOutput(
        loss = loss,
        reco_loss = reco_loss,
        divergence_var_tau=divergence_var_tau,
        divergence_c = divergence_c
        )

        return output

        

    def get_x_ref(self, x, tau_q):
        noise = (torch.rand_like(tau_q) - 1)*0.5 + 0.25
        noise[:,0] = (torch.rand(noise.shape[0]) - 1)*2*np.pi + np.pi
        if self.mode == 'SIM2':
            noise[:,-1] = 0.5*torch.rand(noise.shape[0]) + 0.5
        M_n, params_n = self.get_M(noise)
        x_ref_trans = self.transform(x, M_n, direction='forward')
        return x_ref_trans