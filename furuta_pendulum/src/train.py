import torch
import time
from .plots import *
from .dynamics import *
from .trajectories import *
from .utils import *
from torch.optim.lr_scheduler import LinearLR


def L2_loss(u, v, w=False, dim=(0, 1), param="L2", rescale_loss=False, denom=None):
    # u nominal trajectory
    # v predicted trajectory
    # u and v expected with shape : [time_steps, batch_size , (q1,p1,q2,p1)]
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
    step, epoch_number, horizon_list=[50, 100, 150, 200, 250, 300], switch_steps=[200, 200, 200, 150, 150, 150]
):
    """
    Description:

    Inputs:

    Outpus:

    """
    # throw error if horizon_list and switch_steps not of the same length
    assert len(horizon_list) == len(switch_steps), " horizon_list and switch_steps must have same length"
    
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


def update_loss_weights(step, w, w_list, switch_steps_weights=[300, 300, 300]):
    """
    Description:

    Inputs:

    Outpus:
    """
    assert len(w_list) == len(switch_steps_weights), " w_list and switch_steps must have same length"

    if step < switch_steps_weights[0]:
        w = w_list[0]
        if step == 0:
            print("loss weights :", w)
    elif step < sum(switch_steps_weights):
        for i in range(1, len(switch_steps_weights)):
            if (step >= sum(switch_steps_weights[0:i])) & (step < sum(switch_steps_weights[0 : i + 1])):
                w = w_list[i]
                if step == sum(switch_steps_weights[0:i]):
                    print("loss weights :", w)
    else:
        w = w_list[-1]
    return w


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


def generate_multi_level_list_conf2(length=17, num_lists=4):
    """
    This function generates lists of decreasing size containing which resnets should be active
    for the multilevel strategy

    Example output for default parameters:
    >>>
    >>>
    >>>
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


class Training:
    """
    Training class
    """

    def __init__(self):
        pass

    def _init_model(self, model_name):
        pass

    def _output_training_stats(self, epoch, train_loss, test_loss, t0):
        """
        Output and save training stats every epoch or multiple of epochs
        """

        if epoch % self.print_every == 0 or epoch == self.num_epochs - 1:
            print(
                "[%d/%d]\t train loss: %.4f, test loss: %.4f, t: %2.3f"
                % (epoch, self.num_epochs, train_loss, test_loss, time.time() - t0)
            )

        return

    def _train_step(self):
        """
        basic training step of the model
        """
        return

    def _test_step(self):
        """
        basic training step of the model
        """
        return

    def train(self):
        """
        Training function. Uses the classes variables, and returns stats from the
        training procedure
        """
        return


def train(
    device,
    model,
    Ts,
    train_loader,
    test_loader,
    w,
    grad_clip,
    lr_schedule,
    begin_decay,
    epoch_number,
    resnet_config=False,
    alternating=False,
    horizon=False,
    horizon_type=False,
    horizon_list=[50, 100, 150, 200, 250, 300],
    switch_steps=[200, 200, 200, 150, 150, 150],
    epochs=20,
    loss_type="L2",
    collect_grads=False,
    rescale_loss=False,
    rescale_dims=[1, 1, 1, 1],
):
    """
    Description:

    Inputs:

    Outpus:
    """
    ## TODO : make a function for one epoch
    ## first training steps on smaller amount of time_steps

    lr = 1e-3

    optim = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)  # Adam
    if lr_schedule:
        scheduler = LinearLR(optim, start_factor=1.0, end_factor=0.5, total_iters=epochs - begin_decay)

    logs = {"train_loss": [], "test_loss": [], "grads_preclip": [], "grads_postclip": [], "layer_names": []}

    denom = torch.tensor([1], device=device)
    denom_test = torch.tensor([1], device=device)
    horizon_updated = 1

    for step in range(epochs):

        train_loss = 0
        test_loss = 0
        t1 = time.time()

        if horizon_type == "auto":
            horizon_updated, horizon = select_horizon_list(step, epoch_number, horizon_list, switch_steps)
        elif horizon_type == "constant":
            horizon = horizon

        # increase the model size and initialie the new parameters
        if resnet_config:
            model = multilevel_strategy_update(device, step, model, resnet_config, switch_steps)

        model.train()

        for i_batch, (x, t_eval) in enumerate(train_loader):
            # x is [batch_size, time_steps, (q1,p1,q2,p1,u,g1,g2,g3,g4)]
            # print('xshape',x.shape)
            # print('tshape',t_eval.shape)
            t_eval = t_eval[0, :horizon]

            # calculate (max-min) to rescale the loss function
            if rescale_loss:
                if horizon_updated:
                    _, _, denom = get_maxmindenom(
                        x=x[:, :horizon, :4].permute(1, 0, 2), dim1=(0), dim2=(0), rescale_dims=rescale_dims
                    )

            for i in range(2 if alternating else 1):  # only runs once if alternating = False
                if i == 0 and alternating:  # train only the model approximating G
                    model.freeze_H_net(freeze=True)
                    model.freeze_G_net(freeze=False)
                elif i == 1 and alternating:  # train only the model approximating H
                    model.freeze_H_net(freeze=False)
                    model.freeze_G_net(freeze=True)

                train_x_hat = odeint(model, x[:, 0, :4], t_eval, method="rk4", options=dict(step_size=Ts))
                # print('train_x_hat',train_x_hat.shape)
                # train_x_hat is [time_steps, batch_size, (q1,p1,q2,p1)]

                train_loss_mini = L2_loss(
                    x[:, :horizon, :4].permute(1, 0, 2),
                    train_x_hat[:, :, :4],
                    w,
                    param=loss_type,
                    rescale_loss=rescale_loss,
                    denom=denom,
                )
                # after permute x is [time_steps, batch_size, (q1,p1,q2,p1)]

                # loss(u, v, w = False, dim = (0,1), param = loss_type)
                if (not step % 10) and (i_batch == 0):
                    t_plot = time.time()
                    training_plot(t_eval, train_x_hat[:, :, :4], x[:, :horizon, :4])
                    print("plot time :", time.time() - t_plot)

                train_loss = train_loss + train_loss_mini.item()

                train_loss_mini.backward()
                if collect_grads:
                    layer_names, all_grads_preclip = collect_gradients(model.named_parameters())
                    # print(all_grads_preclip)
                if grad_clip:  # gradient clipping to a norm of 1
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if collect_grads:
                    layer_names, all_grads_postclip = collect_gradients(model.named_parameters())
                    logs["layer_names"].append(layer_names)
                    logs["grads_preclip"].append(all_grads_preclip)
                    logs["grads_postclip"].append(all_grads_postclip)

                optim.step()
                optim.zero_grad()

                if step > begin_decay and lr_schedule:
                    scheduler.step()

                # if (horizon == 'schedule') and do_step:
                #   scheduler.step()

        t2 = time.time()
        train_time = t2 - t1

        model.eval()
        if test_loader:
            if not (step % 10):  # run validation every 10 steps
                for x, t_eval in iter(test_loader):

                    with torch.no_grad():  # we won't need gradients for testing
                        # run test data
                        t_eval = t_eval[0, :horizon]
                        if rescale_loss:
                            if horizon_updated:
                                _, _, denom_test = get_maxmindenom(
                                    x=x[:, :horizon, :4].permute(1, 0, 2), dim1=(0), dim2=(0), rescale_dims=rescale_dims
                                )

                        test_x_hat = odeint(model, x[:, 0, :4], t_eval, method="rk4", options=dict(step_size=Ts))

                        # test_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:],w)

                        test_loss_mini = L2_loss(
                            x[:, :horizon, :4].permute(1, 0, 2),
                            test_x_hat[:horizon, :, :4],
                            w,
                            param=loss_type,
                            rescale_loss=rescale_loss,
                            denom=denom_test,
                        )
                        test_loss = test_loss + test_loss_mini.item()
                test_time = time.time() - t2
                print(
                    "epoch {:4d} | train time {:.2f} | train loss {:8e} | test loss {:8e} | test time {:.2f}  ".format(
                        step, train_time, train_loss, test_loss, test_time
                    )
                )
                logs["test_loss"].append(test_loss)

            else:
                print("epoch {:4d} | train time {:.2f} | train loss {:8e} ".format(step, train_time, train_loss))
        else:
            print("epoch {:4d} | train time {:.2f} | train loss {:8e} ".format(step, train_time, train_loss))

        # logging
        logs["train_loss"].append(train_loss)
    return logs
