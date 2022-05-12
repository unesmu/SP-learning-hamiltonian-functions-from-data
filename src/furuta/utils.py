import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
import torch

from torchdiffeq import odeint_adjoint as odeint_adjoint 
# func must be a nn.Module when using the adjoint method
from torchdiffeq import odeint as odeint

import json
import time as time

from .models import *
from .dynamics import *
from .data import *
from .trajectories import *
def set_device():
    # set device to GPU if available otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # if there is a GPU
        print (f'Available device : {torch.cuda.get_device_name(0)}') 
    else :
        print(device)
    return device


def set_furuta_params(which = 'fake'):

    if which == 'fake':
        # The "fake" set of furuta parameters to have similar magnitudes in q1 p1 q2 p2
        Ts = 0.005
        noise_std = 0.0
        C_q1 = 0.0
        C_q2 = 0.0
        g = 9.81
        Jr = 1*1e-5
        Lr = 0.5
        Mp = 5.0
        Lp = 0.1

    if which == 'real':
        # The set of furuta parameters similar to the real furuta
        Ts = 0.005
        noise_std = 0.0
        C_q1 = 0.0
        C_q2 = 0.0
        g = 9.81
        Jr = 5.72*1e-5
        Lr = 0.085
        Mp = 0.024
        Lp = 0.129
    
    return Ts, noise_std, C_q1, C_q2, g, Jr, Lr, Mp, Lp

def count_parameters(model):
    '''
    from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    Description:

    Inputs:

    Outpus:
      
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(device, hidden_dim=90, nb_hidden_layers=4):
    H_net = MLP(input_dim=4, hidden_dim=hidden_dim, nb_hidden_layers=nb_hidden_layers, output_dim=1, activation='x+sin(x)^2')
    model = Simple_MLP(input_dim=4, H_net=H_net, device=None) 
    model.to(device) 
    return model
    
def load_model_nes_hdnn(device, utype, u_func=None, hidden_dim=90, nb_hidden_layers=4):

    G_net = MLP(input_dim=4, hidden_dim=64, nb_hidden_layers=1, output_dim=2, activation='tanh')
    H_net = MLP(input_dim=4, hidden_dim=90, nb_hidden_layers=4, output_dim=1, activation='x+sin(x)^2')
    model = Nes_HDNN(utype=utype, u_func=u_func, G_net=G_net, H_net=H_net, device=device)

    model.to(device) 

    return model 
    
def load_data_device(device, utype, gtype, init_method, u_func=None, g_func=None, time_steps=40, num_trajectories=10, shuffle = False,
                     coord_type='hamiltonian', proportion=0.5, batch_size=1, 
                     Ts = 0.005, noise_std=0.0, C_q1=0.0, C_q2=0.0, 
                     g = 9.81, Jr = 1*1e-5, Lr = 0.5, Mp = 5.0, Lp = 0.5):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    # create trajectories
    q1, p1, q2, p2, energy, derivatives, t_eval = multiple_trajectories_furuta('cpu', utype, gtype, init_method, time_steps, num_trajectories, u_func, g_func,
                          None, Ts , noise_std, C_q1, C_q2, g, Jr,  Lr,  Mp, Lp)

    q1 = q1.to(device)
    p1 = p1.to(device)
    q2 = q2.to(device)
    p2 = p2.to(device)
    energy = energy.to(device)
    derivatives = derivatives.to(device)
    t_eval = t_eval.to(device)
    if u_func is not None:
        u = u_func(t_eval, utype).to(device)
    else:
        u = torch.zeros_like(t_eval,device=device)
    stds = torch.tensor([q1.std(),p1.std(),q2.std(),p2.std()])
    # dataloader to load data in batches
    train_loader, test_loader = data_loader_furuta(u, q1, p1, q2, p2, energy, derivatives, t_eval, batch_size=batch_size,
                          shuffle = shuffle, proportion = proportion, coord_type=coord_type)
    return train_loader, test_loader, stds

def save_stats(PATH, stats, stats_path):
    with open(stats_path, 'w') as file:
        file.write(json.dumps(stats)) # use `json.loads` to do the reverse  
    return

def read_dict(PATH, stats_path):
    # read the stats txt file
    with open(stats_path) as f:
        data = f.read()

    data = json.loads(data)
    return data