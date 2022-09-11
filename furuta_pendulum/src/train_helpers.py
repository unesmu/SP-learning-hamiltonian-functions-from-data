import torch

def L2_loss(u, v, w=False, dim=(0, 1), param="L2", rescale_loss=False, denom=None):
    """
    Calculate the L2 loss of between u and v
     u and v expected with shape : [time_steps, batch_size , (q1,p1,q2,p1)]

    Input:
        - u (tensor) :  nominal trajectory
        - v (tensor) :  predicted trajectory
        - w (bool or tensor) : either false or a tensor containing the weights
                             to rescale each coordinate 
        - dim (tuple) : dimensions on which to sum the terms 
                      (the time step and batch dimensions)
        - param (string) : type of loss, can be one of : 'L2weighted' or 'L2'
        - rescale_loss (bool) : rescale the loss function using min max scaling
        - denom (None or tensor) : if rescale_loss==True this needs to containg the
                                 pre calculated denominator for min max scaling
                                 why? because it is calculated only when the horizon 
                                 is changed
    Output:
        loss (tensor) : scalar loss

    """

    diff = u - v
    if rescale_loss:
        # min max scaling of u and v using nominal statistics
        # ((u-min)/denom)-((v-min)/denom) = (u-v)/denom
        diff = (diff) / denom

    if param == "L2weighted":
        # mean only over time steps and batch_size
        loss = (((diff).mul(w)).pow(2)).mean(dim=dim).sum()
    elif param == "L2":
        loss = (((diff)).pow(2)).mean(dim=dim).sum()

    return loss


def select_horizon_list(
    step,
    epochs,
    horizon_list=[50, 100, 150, 200, 250, 300],
    switch_steps=[200, 200, 200, 150, 150, 150],
):
    """
    Description:
        Select which horizon should be returned depending on the epoch number (step),
        The horizon list indicates which horizons are going to be taken, and switch_steps
        indications how many epochs one horizon is kept for. For example with the default
        parameters, the model is trained for 200 epochs with horizon=50, and then 200
        other epochs with the horizon 200.
    Inputs:
        step (int) : current epoch 
        epochs (int) : total number of epochs
        horizon_list (list) : horizons with which the model will be trained
        switch_steps (list) : number of epochs per horizon
    Outputs:
        horizon_updated (bool) : indicates whether or not the horizon
                                has been updated this run
        horizon (int) : the new horizon value
    """
    # throw error if horizon_list and switch_steps not of the same length
    assert len(horizon_list) == len(
        switch_steps
    ), " horizon_list and switch_steps must have same length"

    horizon_updated = 0
    if step < switch_steps[0]:
        horizon = horizon_list[0]
        if step == 0:
            print("horizon length :", horizon)
            horizon_updated = 1
    elif step < sum(switch_steps):
        for i in range(1, len(switch_steps)):
            if (step >= sum(switch_steps[0:i])) & (step < sum(switch_steps[0 : i + 1])):
                horizon = horizon_list[i]
                if step == sum(switch_steps[0:i]):
                    print("horizon length :", horizon)
                    horizon_updated = 1
    else:
        horizon = horizon_list[-1]
    return horizon_updated, horizon

def multilevel_strategy_update(device, step, model, resnet_config, switch_steps):
    """ 
    Description:
        Implements how the multilevel models are expanded (number of parameters increased) 
        during training, and how to initialise the new model parameters.
    Inputs:
        device (str) : device on which the model is trained
        step (int) : current epoch
        model (nn.Module) : model that is being trained
        resnet_config (int) : resnet config, one of the folowing numbers:
                                    - 1 : Expanding HNN (and its variants)
                                    - 2 : Interp HNN (and its variants)
        switch_steps (list) :
    Outputs:
        model (nn.Module) : model that is being traing and that was just updated

    """
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
        if step < sum(switch_steps[:1]):
            if step == 0:
                print("Model size increased")
            model.H_net.resblock_list = [0, 16]
            model.H_net.alpha = torch.tensor(
                [1 / len(model.H_net.resblock_list)], device=device
            )
        elif step == sum(switch_steps[:1]) and len(model.H_net.resblocks) >= 4:
            print("Model size increased")
            model.H_net.init_new_resblocks(0, 8, 16)
            model.H_net.resblock_list = [0, 8, 16]
            model.H_net.alpha = torch.tensor(
                [1 / len(model.H_net.resblock_list)], device=device
            )
        elif step == sum(switch_steps[:2]) and len(model.H_net.resblocks) >= 6:
            print("Model size increased")
            model.H_net.init_new_resblocks(0, 4, 8)
            model.H_net.init_new_resblocks(8, 12, 16)
            model.H_net.resblock_list = [0, 4, 8, 12, 16]
            model.H_net.alpha = torch.tensor(
                [1 / len(model.H_net.resblock_list)], device=device
            )
        elif step == sum(switch_steps[:3]) and len(model.H_net.resblocks) >= 8:
            print("Model size increased")
            model.H_net.init_new_resblocks(0, 2, 4)
            model.H_net.init_new_resblocks(4, 6, 8)
            model.H_net.init_new_resblocks(8, 10, 12)
            model.H_net.init_new_resblocks(12, 14, 16)
            model.H_net.resblock_list = [0, 2, 4, 6, 8, 10, 12, 14, 16]
            model.H_net.alpha = torch.tensor(
                [1 / len(model.H_net.resblock_list)], device=device
            )
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
            model.H_net.resblock_list = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
            ]
            model.H_net.alpha = torch.tensor(
                [1 / len(model.H_net.resblock_list)], device=device
            )
    return model


def generate_multi_level_list_conf2(length=17, num_lists=4):
    """
    This function generates lists of decreasing size containing which resnets should be active
    for the multilevel strategy. Used to programatically get the lists for multilevel_strategy_update()

    Example output :
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [0, 2, 4, 6, 8, 10, 12, 14, 16],
        [0, 4, 8, 12, 16],
        [0, 8, 16],
        [0, 16]]

    """
    largest_list = list(range(length))
    all_lists = [largest_list]
    prev_list = largest_list
    for j in range(num_lists):
        new_list = []
        for i, elem in enumerate(prev_list):
            if not i % 2 and i < len(prev_list):
                new_list.append(elem)
            if i == len(prev_list) - 1:
                all_lists.append(new_list)
                prev_list = new_list
    return all_lists