import torch
from .data import *
from .trajectories import *

import os


def simple_pendulum_parameters():
    y0 = None
    noise_std = 0.0
    Ts = 0.05
    C = 0.0
    m = 1
    g = 9.81
    l = 1
    return y0, noise_std, Ts, C, m, g, l


# set all paths and create folders :
def create_paths(PATH, save_suffix, model_name):
    model_path = PATH + "data/" + save_suffix + model_name + "/"
    # stats_path = PATH+'data/'+save_suffix+model_name+'/'
    plot_path = PATH + "data/" + save_suffix + model_name + "/img/"
    # train_loader_path = PATH + 'data/'+save_suffix+model_name+'/datasets/'
    # test_loader_path = PATH + 'data/'+save_suffix+model_name+'/datasets/'

    os.makedirs(model_path, exist_ok=True)
    # os.makedirs(stats_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    # os.makedirs(train_loader_path, exist_ok=True)
    # os.makedirs(test_loader_path, exist_ok=True)

    return model_path, plot_path  # , train_loader_path, test_loader_path # stats_path,


def L2_loss(u, v, w=False, dim=(0, 1), param="L2"):
    # u nominal
    # v approximate
    # u and v expected with shape : [time_steps, batch_size , (q1,p1)]
    if param == "L2":
        # mean only over time steps and batch_size
        loss = (((u - v)).pow(2)).mean(dim=dim).sum()
    elif param == "L2weighted":
        loss = (((u - v).mul(w)).pow(2)).mean(dim=dim).sum()
    return loss  # ((u-v)*w).pow(2).mean()


def load_data_device(
    time_steps,
    num_trajectories,
    device,
    batch_size,
    proportion,
    shuffle,
    Ts,
    y0,
    noise_std,
    C,
    m,
    g,
    l,
    u_func,
    g_func,
    coord_type="hamiltonian",
):
    # create trajectories
    q, p, t_eval, _, _ = multiple_trajectories(
        time_steps=time_steps,
        num_trajectories=num_trajectories,
        device=device,
        Ts=Ts,
        y0=y0,
        noise_std=noise_std,
        C=C,
        m=m,
        g=g,
        l=l,
        u_func=u_func,
        g_func=g_func,
        coord_type=coord_type,
    )

    # dataloader to load data in batches
    train_loader, test_loader = data_loader(q, p, t_eval, batch_size, device, shuffle=shuffle, proportion=proportion)

    return train_loader, test_loader


def select_horizon_list(step, 
                        horizon_list=[50, 100, 150, 200, 250, 300], 
                        switch_steps=[200, 200, 200, 150, 150, 150]
                        ):
    """
    Description:

    Inputs:

    Outpus:

    """
    # throw error if horizon_list and switch_steps not of the same length
    assert len(horizon_list) == len(switch_steps), " horizon_list and switch_steps must have same length"

    if step < switch_steps[0]:
        horizon = horizon_list[0]
        if step == 0:
            print("horizon length :", horizon)
    elif step < sum(switch_steps):
        for i in range(1, len(switch_steps)):
            if (step >= sum(switch_steps[0:i])) & (step < sum(switch_steps[0 : i + 1])):
                horizon = horizon_list[i]
                if step == sum(switch_steps[0:i]):
                    print("horizon length :", horizon)
    else:
        horizon = horizon_list[-1]
    return horizon


def multilevel_strategy_update(device, step, model, resnet_config, switch_steps):
    """ """
    # resnet strategy 1 :
    if resnet_config == 1 or resnet_config == 3:
        if step < sum(switch_steps[:1]):
            if step == 0:
                print("Model size increased")
            model.H_net.resblock_list = [0]
        elif step == sum(switch_steps[:1]) and len(model.H_net.resblocks) >= 2:
            print("Model size increased")
            model.H_net.resblock_list = [0, 1]
        elif step == sum(switch_steps[:2]) and len(model.H_net.resblocks) >= 3:
            print("Model size increased")
            model.H_net.resblock_list = [0, 1, 2]
        elif step == sum(switch_steps[:3]) and len(model.H_net.resblocks) >= 4:
            print("Model size increased")
            model.H_net.resblock_list = [0, 1, 2, 3]
        elif step == sum(switch_steps[:4]) and len(model.H_net.resblocks) >= 5:
            print("Model size increased")
            model.H_net.resblock_list = [0, 1, 2, 3, 4]
        elif step == sum(switch_steps[:5]) and len(model.H_net.resblocks) >= 6:
            print("Model size increased")
            model.H_net.resblock_list = [0, 1, 2, 3, 4, 5]

    # resnet strategy 2 :
    if resnet_config == 2:
        if step < sum(switch_steps[:1]):  # init : [0,1,2,3,4,5,6,7,8,9,10,11]
            if step == 0:
                print("Model size increased")
            model.H_net.resblock_list = [0, 16]
            model.H_net.alpha = torch.tensor([1 / len(model.H_net.resblock_list)], device=device)
        elif step == sum(switch_steps[:1]) and len(model.H_net.resblocks) >= 4:
            print("Model size increased")
            model.H_net.init_new_resblocks(0, 8, 16)
            model.H_net.resblock_list = [0, 8, 16]
            model.H_net.alpha = torch.tensor([1 / len(model.H_net.resblock_list)], device=device)
        elif step == sum(switch_steps[:2]) and len(model.H_net.resblocks) >= 6:
            print("Model size increased")
            model.H_net.init_new_resblocks(0, 4, 8)
            model.H_net.init_new_resblocks(8, 12, 16)
            model.H_net.resblock_list = [0, 4, 8, 12, 16]
            model.H_net.alpha = torch.tensor([1 / len(model.H_net.resblock_list)], device=device)
        elif step == sum(switch_steps[:3]) and len(model.H_net.resblocks) >= 8:
            print("Model size increased")
            model.H_net.init_new_resblocks(0, 2, 4)
            model.H_net.init_new_resblocks(4, 6, 8)
            model.H_net.init_new_resblocks(8, 10, 12)
            model.H_net.init_new_resblocks(12, 14, 16)
            model.H_net.resblock_list = [0, 2, 4, 6, 8, 10, 12, 14, 16]
            model.H_net.alpha = torch.tensor([1 / len(model.H_net.resblock_list)], device=device)
        elif step == sum(switch_steps[:4]) and len(model.H_net.resblocks) >= 10:
            print("Model size increased")
            model.H_net.init_new_resblocks(0, 1, 2)
            model.H_net.init_new_resblocks(2, 3, 4)
            model.H_net.init_new_resblocks(4, 5, 6)
            model.H_net.init_new_resblocks(8, 9, 10)
            model.H_net.init_new_resblocks(10, 11, 12)
            model.H_net.init_new_resblocks(11, 12, 13)
            model.H_net.init_new_resblocks(12, 13, 14)
            model.H_net.init_new_resblocks(14, 15, 16)
            model.H_net.resblock_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            model.H_net.alpha = torch.tensor([1 / len(model.H_net.resblock_list)], device=device)
    return model
