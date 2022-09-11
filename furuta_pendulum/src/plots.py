import matplotlib.pyplot as plt

import torch

from torchdiffeq import odeint as odeint

from .trajectories import *
import time as time

""" FOR DATA """


def plot_traj_furuta(
    t_eval,
    q1,
    p1,
    q2,
    p2,
    energy=torch.tensor(False),
    title="Trajectory of the generalized coordinates",
    coord_type="hamiltonian",
):
    """
    This function plots the generalised variables q1, p1, q2, p2, and the energy
    at the time t_eval at which they were evaluated
    Inputs:
      t_eval (tensor) : vector containing evaluation times of the generalized coordinates
      q1 (tensor) : generalized position q1
      p1 (tensor) : generalized momentum p1
      q2 (tensor) : generalized position q1
      p2 (tensor) : generalized momentum p2
      energy (tensor) : energy evaluated at the provided coordinates, if
          not provided, it will not appear in the plot (default = torch.tensor(False))
    Outputs:
      None
    """
    if torch.any(energy):

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            1, 5, figsize=(15, 4), constrained_layout=True, sharex=True
        )  # , sharey=True)
        ax5.plot(t_eval, energy, label="energy")

        # ax1.legend()
        ax5.set_title("Energy", fontsize=10)
        ax5.set_xlabel("time (s)")
        ax5.set_ylabel("E")
        ax5.set_ylim((0, torch.max(energy) * 1.1))
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            1, 4, figsize=(12, 4), constrained_layout=True, sharex=True
        )  # , sharey=True)

    ax1.plot(t_eval, q1, label="q1")
    ax2.plot(t_eval, p1, label="p1")
    ax3.plot(t_eval, q2, label="q2")
    ax4.plot(t_eval, p2, label="p2")

    # ax1.legend()
    ax1.set_title("generalized position (q1)", fontsize=10)
    ax1.set_xlabel("time[s]")
    ax1.set_ylabel("q1[rad]")

    # ax2.legend()
    ax2.set_title("generalized momentum (p1)", fontsize=10)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("p1")

    ax3.set_title("generalized position (q2)", fontsize=10)
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("q2[rad]")

    ax4.set_title("generalized momentum (p2)", fontsize=10)
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("p2")

    if coord_type == "newtonian":
        ax2.set_title(r"$\dot{q1}[rad/s]$", fontsize=10)
        ax2.set_ylabel(r"$\dot{q1}[rad/s]$")
        ax4.set_title(r"$\dot{q2}[rad/s]$", fontsize=10)
        ax4.set_ylabel(r"$\dot{q2}[rad/s]$")

    fig.suptitle(title, fontsize=12)
    plt.show()
    return


def plot_traj_furuta_withinput(
    t_eval,
    q1,
    p1,
    q2,
    p2,
    energy=None,
    input=None,
    title="Trajectory of the generalized coordinates",
    coord_type="hamiltonian",
):
    """
    This function plots the generalised variables q1, p1, q2, p2, and the energy
    at the time t_eval at which they were evaluated
    Inputs:
      t_eval (tensor) : vector containing evaluation times of the generalized coordinates
      q1 (tensor) : generalized position q1
      p1 (tensor) : generalized momentum p1
      q2 (tensor) : generalized position q1
      p2 (tensor) : generalized momentum p2
      energy (tensor) : energy evaluated at the provided coordinates, if
          not provided, it will not appear in the plot (default = torch.tensor(False))
    Outputs:
      None
    """

    fig, ax = plt.subplots(
        2, 3, figsize=(15, 4), constrained_layout=True, sharex=True
    )
    if energy is not None:
        ax[1, 2].plot(t_eval, energy, label="energy")
        ax[1, 2].set_title("Energy", fontsize=10)
        ax[1, 2].set_xlabel("time (s)")
        ax[1, 2].set_ylabel("E")
        ax[1, 2].set_ylim((0, torch.max(energy) * 1.1))
    else:
        ax[1, 2].set_axis_off()

    ax[0, 2].plot(t_eval, input)
    ax[0, 2].set_title("Input", fontsize=10)
    ax[0, 2].set_xlabel("time (s)")
    ax[0, 2].set_ylabel("U")

    ax[0, 0].plot(t_eval, q1, label="q1")
    ax[0, 0].set_title("generalized position (q1)", fontsize=10)
    ax[0, 0].set_xlabel("time[s]")
    ax[0, 0].set_ylabel("q1[rad]")

    ax[1, 0].plot(t_eval, p1, label="p1")
    ax[1, 0].set_title("generalized momentum (p1)", fontsize=10)
    ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].set_ylabel("p1")

    ax[0, 1].plot(t_eval, q2, label="q2")
    ax[0, 1].set_title("generalized position (q2)", fontsize=10)
    ax[0, 1].set_xlabel("time [s]")
    ax[0, 1].set_ylabel("q2[rad]")

    ax[1, 1].plot(t_eval, p2, label="p2")
    ax[1, 1].set_title("generalized momentum (p2)", fontsize=10)
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_ylabel("p2")

    if coord_type == "newtonian":
        ax[1, 0].set_title(r"$\dot{q1}[rad/s]$", fontsize=10)
        ax[1, 0].set_ylabel(r"$\dot{q1}[rad/s]$")
        ax[1, 1].set_title(r"$\dot{q2}[rad/s]$", fontsize=10)
        ax[1, 1].set_ylabel(r"$\dot{q2}[rad/s]$")

    fig.suptitle(title, fontsize=12)
    plt.show()
    return


""" FOR RESULTS """


def plot_furuta_hat_nom(
    device,
    model,
    u_func,
    g_func,
    utype,
    gtype,
    data_loader_t,
    n,
    t_max,
    C_q1,
    C_q2,
    g,
    Jr,
    Lr,
    Mp,
    Lp,
    t_plot=None,
    show_pred=True,
    only_pred=False,
    H_or_Input="input",
    title="Trajectory of the generalized coordinates",  # , coord_type='hamiltonian'
    file_path=None,
    w_rescale=None,
):
    """
    Description:
      This function plots the generalised variables q p, and the energy at the time
      t_eval at which they were evaluated
    Inputs:
        - device (string) : device to use to generate the trajectories 
                (cpu or GPU, use get_device() )
        - model (nn.Module) : model that has been trained
        - u_func (class) : class which containes the input function
        - g_func (class) : class which containes the input matrix
        - utype (string): can be one of :
                        'chirp'
                        'sine'
                        'tanh'
                        'multisine'
                        'step'
                        None
        - gtype (string or bool): can be one of :
                                    'simple'
                                     None
        - data_loader_t (dataloader object) : dataloader that will be used for the 
                                               nominal trajectories
        - n (int) : integer indicater which sample in the batch from data_loader_tt should
                    be plotted
        - t_max (int) : how many samples were used in training, these will be shown in
                                green in the plot 
        - C_q1 (Float) : friction coefficient
        - C_q2 (Float) : friction coefficient
        - g, Jr, Lr, Mp, Lp (Float) : furuta pendulum parameters
        - t_plot () : time step at which the plot should stop
        - only_pred (bool) : use 1 if using test trajectories
                             use 0 if using train trajectories
        - H_or_Input () : if "input" will plot the input function in a plot,
                        otherwise it will know the Hamiltonian function
        - title (string): title of the plot
        - file_path (string) : where to save the plot
        - w_rescale (list): list containing how the coordinates were rescaled

    Outputs:
      None
    """

    x_nom, t_eval = next(iter(data_loader_t))

    t_eval = t_eval[0, :]  # otherwise it is by batch
    time_steps = len(t_eval)

    if t_plot:
        time_steps = t_plot
        t_eval = t_eval[:t_plot]
    else:
        t_plot = time_steps

    Ts = t_eval[0]

    # predicted trajectory
    x_hat = odeint(model, x_nom[:, 0, :], t_eval, method="rk4").detach()
    x_hat = x_hat.detach()

    # to do: make this concise with torch split or chunck
    q1_hat = x_hat[:, n, 0].unsqueeze(dim=0)
    p1_hat = x_hat[:, n, 1].unsqueeze(dim=0)
    q2_hat = x_hat[:, n, 2].unsqueeze(dim=0)
    p2_hat = x_hat[:, n, 3].unsqueeze(dim=0)

    E_hat, _ = get_energy_furuta(
        device,
        time_steps,
        Ts,
        u_func,
        g_func,
        q1_hat / w_rescale[0],
        p1_hat / w_rescale[1],
        q2_hat / w_rescale[2],
        p2_hat / w_rescale[3],
        C_q1,
        C_q2,
        g,
        Jr,
        Lr,
        Mp,
        Lp,
    )
    H_hat = furuta_H(
        q1_hat / w_rescale[0],
        p1_hat / w_rescale[1],
        q2_hat / w_rescale[2],
        p2_hat / w_rescale[3],
        g,
        Jr,
        Lr,
        Mp,
        Lp,
    )

    # H_hat = model.H_net(x_hat[:,0,:]).detach().squeeze()
    E_hat = E_hat.detach().cpu().squeeze()
    H_hat = H_hat.detach().cpu().squeeze()

    # nominal trajectory
    q1_hat, p1_hat, q2_hat, p2_hat = (
        q1_hat.squeeze(),
        p1_hat.squeeze(),
        q2_hat.squeeze(),
        p2_hat.squeeze(),
    )
    x_nom = x_nom.detach()

    q1_nom = x_nom[n, :t_plot, 0].unsqueeze(dim=0)
    p1_nom = x_nom[n, :t_plot, 1].unsqueeze(dim=0)
    q2_nom = x_nom[n, :t_plot, 2].unsqueeze(dim=0)
    p2_nom = x_nom[n, :t_plot, 3].unsqueeze(dim=0)

    E_nom, _ = get_energy_furuta(
        device,
        time_steps,
        Ts,
        u_func,
        g_func,
        q1_nom / w_rescale[0],
        p1_nom / w_rescale[1],
        q2_nom / w_rescale[2],
        p2_nom / w_rescale[3],
        C_q1,
        C_q2,
        g,
        Jr,
        Lr,
        Mp,
        Lp,
        time_=t_eval,
    )
    H_nom = furuta_H(
        q1_nom / w_rescale[0],
        p1_nom / w_rescale[1],
        q2_nom / w_rescale[2],
        p2_nom / w_rescale[3],
        g,
        Jr,
        Lr,
        Mp,
        Lp,
    )

    E_nom = E_nom.detach().cpu().squeeze()
    H_nom = H_nom.detach().cpu().squeeze()
    q1_nom, p1_nom, q2_nom, p2_nom = (
        q1_nom.squeeze().cpu(),
        p1_nom.squeeze().cpu(),
        q2_nom.squeeze().cpu(),
        p2_nom.squeeze().cpu(),
    )
    t_eval = t_eval.detach().cpu()
    fig, ax = plt.subplots(
        2, 3, figsize=(15, 4), constrained_layout=True, sharex=True
    ) 

    if H_or_Input == "input":
        H_nom = u_func.forward(t_eval.to(device))
        ax[1, 2].set_title("Input", fontsize=10)
        ax[1, 2].set_xlabel("time (s)")
        ax[1, 2].set_ylabel("u")
        t_eval = t_eval = t_eval.detach().cpu()

    else:
        ax[1, 2].set_title("Hamiltonian", fontsize=10)
        ax[1, 2].set_xlabel("time (s)")
        ax[1, 2].set_ylabel("H")
    for q1, p1, q2, p2, E, H, label in [
        [q1_nom, p1_nom, q2_nom, p2_nom, E_nom, H_nom, "nominal"]
    ]:

        label_train = "train"
        color1 = "g"
        if only_pred:
            color1 = "C0"
        ax[0, 0].plot(t_eval[:t_max], q1[:t_max], label=label_train, c=color1)
        ax[0, 0].plot(t_eval[t_max - 1 :], q1[t_max - 1 :], label=label, c="C0")

        ax[1, 0].plot(t_eval[:t_max], p1[:t_max], label=label_train, c=color1)
        ax[1, 0].plot(t_eval[t_max - 1 :], p1[t_max - 1 :], label=label, c="C0")

        ax[0, 1].plot(t_eval[:t_max], q2[:t_max], label=label_train, c=color1)
        ax[0, 1].plot(t_eval[t_max - 1 :], q2[t_max - 1 :], label=label, c="C0")

        ax[1, 1].plot(t_eval[:t_max], p2[:t_max], label=label_train, c=color1)
        ax[1, 1].plot(t_eval[t_max - 1 :], p2[t_max - 1 :], label=label, c="C0")

        ax[0, 2].plot(t_eval[:t_max], E[:t_max], label=label_train, c=color1)
        ax[0, 2].plot(t_eval[t_max - 1 :], E[t_max - 1 :], label=label, c="C0")
        ax[1, 2].plot(t_eval[:], H[:], label=label_train, c="C0")
    for q1, p1, q2, p2, E, H, label in [
        [q1_hat, p1_hat, q2_hat, p2_hat, E_hat, H_hat, "prediction"]
    ]:

        q1 = q1.cpu()
        p1 = p1.cpu()
        q2 = q2.cpu()
        p2 = p2.cpu()
        E = E.cpu()
        H = H.cpu()
        label_train = "prediction"
        color = "r"
        if only_pred:
            t_max = time_steps
            color = "r"
            label_train = "prediction"
            show_pred = False

        ax[0, 0].plot(t_eval[:], q1[:], label=label_train, c=color, linewidth=1)

        ax[1, 0].plot(t_eval[:], p1[:], label=label_train, c=color, linewidth=1)

        ax[0, 1].plot(t_eval[:], q2[:], label=label_train, c=color, linewidth=1)

        ax[1, 1].plot(t_eval[:], p2[:], label=label_train, c=color, linewidth=1)

        ax[0, 2].plot(t_eval[:], E[:], label=label_train, c=color, linewidth=1)

        if H_or_Input == "H":
            ax[1, 2].plot(t_eval[:], H[:t_max], label=label_train, c=color)

    ax[0, 0].legend()

    # add labels and titles on every plot
    ax[0, 0].set_title("generalized position (q1)", fontsize=10)
    ax[0, 0].set_xlabel("time[s]")
    ax[0, 0].set_ylabel("q1[rad]")

    ax[1, 0].set_title("generalized momentum (p1)", fontsize=10)
    ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].set_ylabel("p1")

    ax[0, 1].set_title("generalized position (q2)", fontsize=10)
    ax[0, 1].set_xlabel("time [s]")
    ax[0, 1].set_ylabel("q2[rad]")

    ax[1, 1].set_title("generalized momentum (p2)", fontsize=10)
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_ylabel("p2")

    ax[0, 2].set_title("Energy", fontsize=10)
    ax[0, 2].set_xlabel("time (s)")
    ax[0, 2].set_ylabel("E")

    # add larger title on top
    fig.suptitle(title, fontsize=12)

    if file_path is not None:
        plt.savefig(file_path, format="png", dpi=400)
    plt.show()
    return

def train_test_loss_plot(
    loss_train,
    loss_test,
    epochs,
    file_path=None,
    horizons=[100, 150, 200, 250, 300],
    switch_steps=[200, 200, 200, 200, 200],
    title="train and test loss per epoch",
):
    """
    Description:
        plot of train and test trajectories after training
    Inputs:
         - loss_train (tensor) : train loss vector
         - loss_test (tensor) : test loss vector
         - epochs (int) : maximum number of training epochs
         - file_path (string) : path where the plot should be saved
         - horizons (list) : horizons with which the model will be trained
         - switch_steps (list) : number of epochs per horizon
         - title (string) : title of the plot
    Outputs:
        None
    """
    # convert switch steps from : [200,200,200,200,200] to [200,400,600...]
    horizon_steps = []
    horizon_steps.append(0)
    for i, number in enumerate(switch_steps):
        horizon_steps.append(horizon_steps[i] + number)
    horizon_steps = horizon_steps[1:-1]

    fig, ax = plt.subplots(figsize=(10, 4))


    plt.plot(epochs, loss_train, label="train")

    if not loss_test == []:  # if loss_test exists
        plt.plot(epochs[::10], loss_test, label="test")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.title(title)

    if horizons:
        for i, epoch_num in enumerate(horizon_steps[:-1]):
            ax.annotate(
                "%3d" % horizons[i],
                xy=(epochs[epoch_num], loss_train[epoch_num]),
                xycoords="data",
                xytext=(-20, 50),
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="angle,angleA=0,angleB=90,rad=10",
                ),
            )
        ax.annotate(
            "%3d" % horizons[-1],
            xy=(epochs[horizon_steps[-1]], loss_train[horizon_steps[-1]]),
            xycoords="data",
            xytext=(-20, 50),
            textcoords="offset points", 
            arrowprops=dict(
                arrowstyle="->",

                connectionstyle="angle,angleA=0,angleB=90,rad=10",
            ),
        )

    if file_path is not None:
        plt.savefig(file_path, format="png", dpi=400)
    plt.show()

    return


def training_plot(t_eval, train_x, nominal_x):
    """
    Plot of nominal vs predicted trajectories used during training

    Input 
        - t_eval (tensor) : time at which the coordinates were evaluated
        - train_x (tensor) : predicted trajectory
        - nominal_x (tensor) : nominal trajectory
    Output
        None
    """

    # train_x is [batch_size,(q1,p1,q2,p1),time_steps]
    # nominal_x is [time_steps, batch_size, (q1,p1,q2,p1)]

    fig, ax = plt.subplots(
        2, 2, figsize=(10, 5), constrained_layout=True, sharex=True
    ) 
    t_eval_cpu = t_eval.detach().cpu()

    ax[0, 0].plot(t_eval_cpu, train_x[:, 0, 0].detach().cpu(), label="train", c="g")
    ax[1, 0].plot(t_eval_cpu, train_x[:, 0, 1].detach().cpu(), label="train", c="g")
    ax[0, 1].plot(t_eval_cpu, train_x[:, 0, 2].detach().cpu(), label="train", c="g")
    ax[1, 1].plot(t_eval_cpu, train_x[:, 0, 3].detach().cpu(), label="train", c="g")

    ax[0, 0].plot(t_eval_cpu, nominal_x[0, :, 0].detach().cpu(), label="nominal")
    ax[1, 0].plot(t_eval_cpu, nominal_x[0, :, 1].detach().cpu(), label="nominal")
    ax[0, 1].plot(t_eval_cpu, nominal_x[0, :, 2].detach().cpu(), label="nominal")
    ax[1, 1].plot(t_eval_cpu, nominal_x[0, :, 3].detach().cpu(), label="nominal")

    ax[0, 0].set_title("generalized position (q1)", fontsize=10)
    ax[0, 0].set_xlabel("time[s]")
    ax[0, 0].set_ylabel("q1[rad]")

    ax[1, 0].set_title("generalized momentum (p1)", fontsize=10)
    ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].set_ylabel("p1")

    ax[0, 1].set_title("generalized position (q2)", fontsize=10)
    ax[0, 1].set_xlabel("time [s]")
    ax[0, 1].set_ylabel("q2[rad]")

    ax[1, 1].set_title("generalized momentum (p2)", fontsize=10)
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_ylabel("p2")
    ax[1, 1].legend()

    # add larger title on top
    fig.suptitle("intermediate plot of trajectories", fontsize=12)

    plt.show()
    return


def plot_grads(stats, file_path, save):
    """ 
    Description:
        Plot the gradient norm in multiple layers of a neural network depending
        on the epoch
    Input:
        - stats (dict) : dict containing stats saved from an experiment where
                        gradients where logged
        - file_path (int) : path where to save this plot
        - save (int) : whether or not to save this plot

    Output:
        None
    """
    grads_preclip_min = [[] for _ in range(len(stats["layer_names"][0]))]
    grads_preclip_max = [[] for _ in range(len(stats["layer_names"][0]))]
    grads_preclip_mean = [[] for _ in range(len(stats["layer_names"][0]))]
    grads_postclip_min = [[] for _ in range(len(stats["layer_names"][0]))]
    grads_postclip_max = [[] for _ in range(len(stats["layer_names"][0]))]
    grads_postclip_mean = [[] for _ in range(len(stats["layer_names"][0]))]

    for layer in range(len(stats["grads_preclip"][0])):
        for iteration in range(len(stats["grads_preclip"])):
            grads_preclip_min[layer].append(
                stats["grads_preclip"][iteration][layer].abs().min()
            )
            grads_preclip_max[layer].append(
                stats["grads_preclip"][iteration][layer].abs().max()
            )
            grads_preclip_mean[layer].append(
                stats["grads_preclip"][iteration][layer].abs().mean()
            )
            grads_postclip_min[layer].append(
                stats["grads_postclip"][iteration][layer].abs().min()
            )
            grads_postclip_max[layer].append(
                stats["grads_postclip"][iteration][layer].abs().max()
            )
            grads_postclip_mean[layer].append(
                stats["grads_postclip"][iteration][layer].abs().mean()
            )
    grads_preclip_min = torch.tensor(grads_preclip_min)
    grads_preclip_max = torch.tensor(grads_preclip_max)
    grads_preclip_mean = torch.tensor(grads_preclip_mean)
    grads_postclip_min = torch.tensor(grads_postclip_min)
    grads_postclip_max = torch.tensor(grads_postclip_max)
    grads_postclip_mean = torch.tensor(grads_postclip_mean)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    plt.yscale("log")
    for i in range(2, grads_preclip_min.shape[0]):
        ax[0].plot(grads_preclip_mean[i, :], label=stats["layer_names"][0][i])
    for i in range(2, grads_preclip_min.shape[0]):
        ax[1].plot(grads_postclip_mean[i, :])

    ax[0].set_title("before clipping")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("mean gradient value")
    ax[0].set_yscale("log")

    ax[1].set_title("after clipping")
    ax[1].set_xlabel("iteration")
    ax[1].set_yscale("log")

    fig.suptitle("Mean gradient value in each layer")
    fig.legend(loc=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(right=0.70)
    if save:
        plt.savefig(file_path, format="png", dpi=400)
    plt.show()
