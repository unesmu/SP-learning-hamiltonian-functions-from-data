import torch
import random
import json
import numpy as np

from .models import *
from .dynamics import *
from .data import *
from .trajectories import *


def collect_gradients(named_parameters):
    """
    Collect the gradients of all the named layers in a model

    Call after loss.backwards():
    "collect_gradients(self.model.named_parameters())" to collect gradients
    """

    all_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            all_grads.append(p.grad.detach().clone())

    return layers, all_grads


def set_device():
    """
    Chooses a computing device, GPU if available otherwise it will use CPU
    """
    # set device to GPU if available otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # if there is a GPU
        print(f"Available device : {torch.cuda.get_device_name(0)}")
    else:
        print(device)
    return device


def set_furuta_params(which="fake"):
    """
    Set the parameters that describe the dynamics of the furuta pendulum either to 
    real ones (that are close to the quobe servo) or fake ones
    """
    if which == "fake":
        # The "fake" set of furuta parameters to have similar magnitudes in q1 p1 q2 p2
        Ts = 0.005
        noise_std = 0.0
        C_q1 = 0.0
        C_q2 = 0.0
        g = 9.81
        Jr = 1 * 1e-5
        Lr = 0.5
        Mp = 5.0
        Lp = 0.1

    if which == "real":
        # The set of furuta parameters similar to the real furuta
        Ts = 0.005
        noise_std = 0.0
        C_q1 = 0.0
        C_q2 = 0.0
        g = 9.81
        Jr = 5.72 * 1e-5
        Lr = 0.085
        Mp = 0.024
        Lp = 0.129

    return Ts, noise_std, C_q1, C_q2, g, Jr, Lr, Mp, Lp


def count_parameters(model):
    """
    Returns the number of learnable parameters in a model
    Inputs:
        model(nn.Module): the model for which the number of parameters are desired
    Outpus:
        _ (int): number of learnable parameters in the model
    Source : https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_stats(dict, dict_path):
    """
    Save dictionary file to disk (to a .txt)
    """
    with open(dict_path, "w") as file:
        file.write(json.dumps(dict))
    return

def read_dict(dict_path):
    """
    Read a dictionary file  (from a .txt)
    """
    with open(dict_path) as f:
        dict = f.read()
    dict = json.loads(dict)
    return dict


def get_maxmindenom(x, dim1, dim2, rescale_dims):
    """
    Given a tensor x that has 3 dimensions, calculate 
    the minimum and maximum on one dimension, and return
    them along with denom which would be the denominator in
    a min max formula
    """
    maximums = x.amax(dim=dim1).unsqueeze(dim=dim2)
    minimums = x.amin(dim=dim1).unsqueeze(dim=dim2)
    denom = (maximums - minimums).abs()
    denom[:, :, ~(torch.Tensor(rescale_dims).bool())] = 1  #
    print("min max values updated")
    return maximums, minimums, denom


def name_from_params(
    Ts,
    rescale_loss,
    weights,
    epoch_number,
    num_params,
    utype,
    model_name,
    num_trajectories,
    furuta_type,
    noise_std,
    grad_clip,
    lr_schedule,
    C_q1,
    C_q2,
    horizon,
    min_max_rescale,
    w_rescale=None,
):
    """
    Function used to create the name of the directory 
    for the model from the model's name and parameters used for training
    """
    save_prefix = "{:d}e_p{:d}k_Ts{:1.3f}_".format(
        epoch_number, int((num_params - num_params % 1000) / 1000), Ts
    )
    if utype is None:
        input = "noinput"
    else:
        input = utype
    save_prefix = (
        model_name
        + "_"
        + input
        + "_"
        + str(num_trajectories)
        + "traj"
        + "_"
        + furuta_type
        + "_"
        + "noise"
        + str(noise_std)
        + "_"
        + save_prefix
    )
    if grad_clip:
        save_prefix = save_prefix + "gradcl_"
    if lr_schedule:
        save_prefix = save_prefix + "lrsched_"
    if C_q1 == 0 and C_q2 == 0:
        save_prefix = save_prefix + "nodissip_"
    else:
        save_prefix = save_prefix + "wdissip_"
    if horizon:
        save_prefix = save_prefix + "constanthorizon_"
    if rescale_loss:
        save_prefix = save_prefix + "rescaledloss_"
    if min_max_rescale:
        save_prefix = save_prefix + "trajminmaxrescale_"
    if w_rescale is not None:
        if w_rescale is not [1, 1, 1, 1]:
            save_prefix = save_prefix + "stdrescale_"
    return save_prefix


def is_same_size(horizon_list, switch_steps):
    """
    To check if two lists are the same length
    """
    if len(horizon_list) == len(switch_steps):
        print("horizon_list and switch_steps have the same size")
    else:
        raise ValueError("horizon_list and switch_steps do NOT have the same size")


def set_all_seeds(manualSeed=123, new_results=False):
    """
    Set random seed for reproducibility
    """
    if new_results:
        manualSeed = random.randint(1, 10000)  # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
