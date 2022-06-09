import torch

from torchdiffeq import odeint_adjoint as odeint_adjoint
# func must be a nn.Module when using the adjoint method
from torchdiffeq import odeint as odeint
import random
import json
import numpy as np

from .models import *
from .dynamics import *
from .data import *
from .trajectories import *


def collect_gradients(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    all_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            all_grads.append(p.grad.detach().clone())

    return layers, all_grads


def set_device():
    # set device to GPU if available otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # if there is a GPU
        print(f'Available device : {torch.cuda.get_device_name(0)}')
    else:
        print(device)
    return device


def set_furuta_params(which='fake'):

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
    H_net = MLP(input_dim=4, hidden_dim=hidden_dim,
                nb_hidden_layers=nb_hidden_layers, output_dim=1, activation='x+sin(x)^2')
    model = simple_HNN(input_dim=4, H_net=H_net, device=None)
    model.to(device)
    return model


def load_model_nes_hdnn(device, utype, u_func=None, hidden_dim=90, nb_hidden_layers=4):
    G_net = MLP(input_dim=4, hidden_dim=64, nb_hidden_layers=1,
                output_dim=2, activation='tanh')
    H_net = MLP(input_dim=4, hidden_dim=90, nb_hidden_layers=4,
                output_dim=1, activation='x+sin(x)^2')
    model = Input_HNN(utype=utype, u_func=u_func,
                     G_net=G_net, H_net=H_net, device=device)
    model.to(device)
    return model


def load_data_device(device, init_method, w_rescale, u_func=None, g_func=None, time_steps=40, num_trajectories=10, shuffle=False,
                     coord_type='hamiltonian', proportion=0.5, batch_size=1,
                     Ts=0.005, noise_std=0.0, C_q1=0.0, C_q2=0.0,
                     g=9.81, Jr=1*1e-5, Lr=0.5, Mp=5.0, Lp=0.5, min_max_rescale=False, rescale_dims=[1, 1, 1, 1]):
    '''
    Description:

    Inputs:

    Outpus:
    '''
    # create trajectories
    q1, p1, q2, p2, energy, derivatives, t_eval = multiple_trajectories_furuta('cpu', init_method, time_steps, num_trajectories, u_func, g_func,
                                                                               None, Ts, noise_std, C_q1, C_q2, g, Jr,  Lr,  Mp, Lp)  # u, G,

    q1 = (q1*w_rescale[0]).detach().to(device)
    p1 = (p1*w_rescale[1]).detach().to(device)
    q2 = (q2*w_rescale[2]).detach().to(device)
    p2 = (p2*w_rescale[3]).detach().to(device)

    if min_max_rescale:
        if rescale_dims[0]:
            q1 = (q1-q1.amin(dim=(1)).unsqueeze(dim=1)) / \
                ((q1.amax(dim=(1))-q1.amin(dim=(1))).abs().unsqueeze(dim=1))
        if rescale_dims[1]:
            q2 = (q2-q2.amin(dim=(1)).unsqueeze(dim=1)) / \
                ((q2.amax(dim=(1))-q2.amin(dim=(1))).abs().unsqueeze(dim=1))
        if rescale_dims[2]:
            p1 = (p1-p1.amin(dim=(1)).unsqueeze(dim=1)) / \
                ((p1.amax(dim=(1))-p1.amin(dim=(1))).abs().unsqueeze(dim=1))
        if rescale_dims[3]:
            p2 = (p2-p2.amin(dim=(1)).unsqueeze(dim=1)) / \
                ((p2.amax(dim=(1))-p2.amin(dim=(1))).abs().unsqueeze(dim=1))

    energy = energy.detach().to(device)
    derivatives = derivatives.detach().to(device)
    t_eval = t_eval.detach().to(device)

    # dataloader to load data in batches
    train_loader, test_loader = data_loader_furuta(q1, p1, q2, p2, energy, derivatives, t_eval, batch_size=batch_size,
                                                   shuffle=shuffle, proportion=proportion, coord_type=coord_type)  # u, G,
    return train_loader, test_loader  


def save_stats(stats, stats_path):
    with open(stats_path, 'w') as file:
        file.write(json.dumps(stats))  
    return


def read_dict(stats_path):
    # read the stats txt file
    with open(stats_path) as f:
        data = f.read()
    data = json.loads(data)
    return data


def get_maxmindenom(x, dim1, dim2, rescale_dims):
    maximums = x.amax(dim=dim1).unsqueeze(dim=dim2)
    minimums = x.amin(dim=dim1).unsqueeze(dim=dim2)
    denom = (maximums - minimums).abs()
    denom[:, :, ~(torch.Tensor(rescale_dims).bool())] = 1  #
    print("min max values updated")
    return maximums, minimums, denom


def name_from_params(Ts, rescale_loss, weights, epoch_number, num_params, utype,
                     model_name, num_trajectories, furuta_type,
                     noise_std, grad_clip, lr_schedule, C_q1, C_q2, horizon, min_max_rescale,w_rescale=None):
    


    weights_title = ' | weights = ' + str(weights)
    save_prefix = '{:d}e_p{:d}k_Ts{:1.3f}_'.format(
        epoch_number, int((num_params-num_params % 1000)/1000), Ts)
    if utype is None:
        input = 'noinput'
    else:
        input = utype
    save_prefix = model_name + '_' + input + '_' + \
        str(num_trajectories)+'traj'+'_' + furuta_type + \
        '_' + 'noise'+str(noise_std)+'_' + save_prefix
    if grad_clip:
        save_prefix = save_prefix + 'gradcl_'
    if lr_schedule:
        save_prefix = save_prefix + 'lrsched_'
    if C_q1 == 0 and C_q2 == 0:
        save_prefix = save_prefix + 'nodissip_'
    else:
        save_prefix = save_prefix + 'wdissip_'
    if horizon:
        save_prefix = save_prefix + 'constanthorizon_'
    if rescale_loss:
        save_prefix = save_prefix + 'rescaledloss_'
    if min_max_rescale:
        save_prefix = save_prefix + 'trajminmaxrescale_'
    if w_rescale is not None :
        if w_rescale is not [1,1,1,1]:
            save_prefix = save_prefix + 'stdrescale_'
    return save_prefix

def is_same_size(horizon_list, switch_steps):
    if len(horizon_list) == len(switch_steps):
        print('horizon_list and switch_steps have the same size')
    else:
        raise ValueError('horizon_list and switch_steps do NOT have the same size')

def set_all_seeds(manualSeed = 123, new_results=False):
    # Set random seed for reproducibility
    if new_results:
        manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
