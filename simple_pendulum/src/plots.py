import torch
import matplotlib.pyplot as plt

from .trajectories import *


def plot_traj_pend(
    t_eval,
    q,
    p,
    energy=torch.tensor(False),
    input=torch.tensor(False),
    coord_type="hamiltonian",
    title="Trajectory of the generalized coordinates",
    save_plot=False,
    file_path=None,
):
    """
    This function plots the generalised variable q p in function of the evaluation
    time. it can also
    Inputs:
      t_eval ()
      q ()
      p ()
    Outputs:
      None
    """
    suptile_fontsize = 20
    small_title_fontsize = 16
    x_fontsize = 15
    y_fontsize = 15
    if torch.any(energy):
        if torch.any(input):
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
            ax3.plot(t_eval, energy, label="energy")

            ax3.set_title("Energy", fontsize=small_title_fontsize)
            ax3.set_xlabel("time (s)", fontsize=x_fontsize)
            ax3.set_ylabel("E", fontsize=y_fontsize)
            ax3.set_ylim((0, torch.max(energy) * 1.1))

            ax4.plot(t_eval, input, label="energy")
            ax4.set_title("Input", fontsize=small_title_fontsize)
            ax4.set_xlabel("time (s)", fontsize=x_fontsize)
            ax4.set_ylabel("U", fontsize=y_fontsize)

        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            ax3.plot(t_eval, energy, label="energy")

            ax3.set_title("Energy", fontsize=small_title_fontsize)
            ax3.set_xlabel("time (s)", fontsize=x_fontsize)
            ax3.set_ylabel("E", fontsize=y_fontsize)
            ax3.set_ylim((0, torch.max(energy) * 1.1))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True, sharex=True)

    ax1.plot(t_eval, q, label="q")
    ax2.plot(t_eval, p, label="p")

    ax1.set_title("generalized position (q)", fontsize=small_title_fontsize)
    ax1.set_xlabel("time (s)", fontsize=x_fontsize)
    ax1.set_ylabel("q", fontsize=y_fontsize)

    ax2.set_title("generalized momentum (p)", fontsize=small_title_fontsize)
    ax2.set_xlabel("time (s)", fontsize=x_fontsize)
    ax2.set_ylabel("p", fontsize=y_fontsize)

    fig.suptitle(title, fontsize=suptile_fontsize)

    if coord_type == "newtonian":
        ax2.set_title(r"$\dot{q}[rad/s]$", fontsize=10)
        ax2.set_ylabel(r"$\dot{q}[rad/s]$")
    if save_plot and (file_path is not None):
        plt.savefig(file_path, format="png", dpi=400)
    plt.show()
    return


def train_test_loss_plot(
    loss_train,
    loss_test,
    epochs,
    test_epochs,
    file_path=None,
    horizons=[100, 150, 200, 250, 300],
    horizon_steps=[200, 400, 550, 700, 850],
    title="train and test loss per epoch",
):
    """
    Description:

    Inputs:

    Outpus:

    """
    fig, ax = plt.subplots(figsize=(10, 4))

    plt.plot(epochs, loss_train, label="train")

    if not loss_test == []:  # if loss_test exists
        plt.plot(test_epochs, loss_test, label="test")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.title(title)

    if horizons:
        for i, epoch_num in enumerate(horizon_steps[:-1]):
            ax.annotate(
                "horizon = %d" % horizons[i],
                xy=(epochs[epoch_num], loss_train[epoch_num]),
                xycoords="data",
                xytext=(-70, 100),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"),
            )
        ax.annotate(
            "horizon = %d" % horizons[-1],
            xy=(epochs[horizon_steps[-1]], loss_train[horizon_steps[-1]]),
            xycoords="data",
            xytext=(+20, 100),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"),
        )

    if file_path is not None:

        plt.savefig(file_path, format="png", dpi=400)

    plt.show()

    return


def plot_results(
    model_training,
    n,
    train_horizon,
    file_path,
    show_pred=True,
    train=True,
    title="Trajectory of the generalized coordinates",
    coord_type="hamiltonian",
    only_pred=False,
):
    if train:
        data_loader_ = model_training.train_loader
    else:
        data_loader_ = model_training.test_loader

    x, t_eval = next(iter(data_loader_))

    t_eval = t_eval[0, :]
    test_x_hat = odeint(model_training.model, x[n : n + 1, 0, :], t_eval, method="rk4")

    energy_nom, _ = get_energy_pendulum(
        t_eval,
        model_training.u_func,
        model_training.g_func,
        x[n, :, 0],
        x[n, :, 1],
        model_training.C,
        model_training.m,
        model_training.g,
        model_training.l,
    )
    energy_pred, _ = get_energy_pendulum(
        t_eval,
        model_training.u_func,
        model_training.g_func,
        test_x_hat[:, 0, 0],
        test_x_hat[:, 0, 1],
        model_training.C,
        model_training.m,
        model_training.g,
        model_training.l,
    )
    input = model_training.u_func.forward(t_eval).cpu().detach()
    t_eval = t_eval.cpu().detach()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4), constrained_layout=True)

    fig.suptitle(title, fontsize=12)

    for q, p, energy, label, linewidth, color in [
        (x[n, :, 0], x[n, :, 1], energy_nom, "nominal", 2, "C0"),
        (test_x_hat[:, 0, 0], test_x_hat[:, 0, 1], energy_pred, "predicted", 1, "r"),
    ]:
        q = q.cpu().detach()
        p = p.cpu().detach()
        energy = energy.cpu().detach()

        ax1.plot(t_eval, q, label=label, color=color, linewidth=linewidth)
        ax3.plot(t_eval, energy, label=label, color=color, linewidth=linewidth)
        ax2.plot(t_eval, p, label=label, color=color, linewidth=linewidth)

        if only_pred == False and label == "nominal":
            ax1.plot(t_eval[:train_horizon], q[:train_horizon], label="train", color="g")
            ax2.plot(t_eval[:train_horizon], p[:train_horizon], label="train", color="g")
            ax3.plot(t_eval[:train_horizon], energy[:train_horizon], label="train", color="g")

        ax1.legend()
        ax1.set_title("generalized position (q)", fontsize=10)
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("q")

        ax2.set_title("generalized momentum (p)", fontsize=10)
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("p")

        ax3.set_title("Energy", fontsize=10)
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel("E")
        ax3.set_ylim((0, torch.max(energy) * 1.1))

    ax4.plot(t_eval, input, label="input")
    ax4.set_title("Input", fontsize=10)
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("U")

    if coord_type == "newtonian":
        ax2.set_title(r"$\dot{q}[rad/s]$", fontsize=10)
        ax2.set_ylabel(r"$\dot{q}[rad/s]$")

    if file_path is not None:
        plt.savefig(file_path, format="png", dpi=400)

    plt.show()
    return
