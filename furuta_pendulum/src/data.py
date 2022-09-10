from torch.utils.data import Dataset, DataLoader, random_split
import torch

from .models import *
from .dynamics import *
from .data import *
from .trajectories import *


class TrajectoryDataset_furuta(Dataset):
    """
    Description:

    Inputs:

    Outpus:

    """

    def __init__(self, q1, p1, q2, p2, t_eval, derivatives, coord_type="hamiltonian"):
        self.t_eval = t_eval
        self.coord_type = coord_type
        self.q1 = q1  # [num_trajectories, time_steps]
        self.p1 = p1  # [num_trajectories, time_steps]
        self.q2 = q2  # [num_trajectories, time_steps]
        self.p2 = p2  # [num_trajectories, time_steps]

        if len(derivatives.shape) == 3:
            # [num_trajectories, time_steps, (dq1/dt,dp1/dt,dq2/dt,dp2/dt)]
            self.dq1dt = derivatives[:, :, 0]
            self.dq2dt = derivatives[:, :, 2]
        else:
            # [num_trajectories, time_steps, (dq1/dt,dp1/dt,dq2/dt,dp2/dt)]
            self.dq1dt = derivatives[:, 0]
            self.dq2dt = derivatives[:, 2]

    def __len__(self):
        return len(self.q1)

    def __getitem__(self, idx):

        if self.coord_type == "hamiltonian":

            q1 = self.q1[idx]
            p1 = self.p1[idx]
            q2 = self.q2[idx]
            p2 = self.p2[idx]

            x = torch.stack((q1, p1, q2, p2), dim=1)

        if self.coord_type == "newtonian":

            q1 = self.q1[idx]
            q2 = self.q2[idx]
            dq1dt = self.dq1dt[idx]
            dq2dt = self.dq2dt[idx]

            x = torch.stack((q1, dq1dt, q2, dq2dt), dim=1)

        t_eval = self.t_eval

        return x, t_eval


def data_loader_furuta(
    q1,
    p1,
    q2,
    p2,
    energy,
    derivatives,
    t_eval,
    batch_size,
    shuffle=True,
    proportion=0.5,
    coord_type="hamiltonian",
):
    """
    Description:

    Inputs:

    Outpus:

    """
    # split  into train and test
    full_dataset = TrajectoryDataset_furuta(
        q1, p1, q2, p2, t_eval, derivatives, coord_type=coord_type
    )
    if proportion:

        train_size = int(proportion * len(full_dataset))
        test_size = len(full_dataset) - train_size

        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        test_loader = DataLoader(test_dataset, batch_size, shuffle)
    else:
        # if proportion is set to None don't split the dataset
        train_dataset = full_dataset
        test_loader = None

    # create the dataloader object from the custom dataset
    train_loader = DataLoader(train_dataset, batch_size, shuffle)

    return train_loader, test_loader


def load_data_device(
    device,
    init_method,
    w_rescale,
    u_func=None,
    g_func=None,
    time_steps=40,
    num_trajectories=10,
    shuffle=False,
    coord_type="hamiltonian",
    proportion=0.5,
    batch_size=1,
    Ts=0.005,
    noise_std=0.0,
    C_q1=0.0,
    C_q2=0.0,
    g=9.81,
    Jr=1 * 1e-5,
    Lr=0.5,
    Mp=5.0,
    Lp=0.5,
    min_max_rescale=False,
    rescale_dims=[1, 1, 1, 1],
):
    """
    Description:

    Inputs:

    Outpus:
    """
    # create trajectories
    q1, p1, q2, p2, energy, derivatives, t_eval = multiple_trajectories_furuta(
        "cpu",
        init_method,
        time_steps,
        num_trajectories,
        u_func,
        g_func,
        None,
        Ts,
        noise_std,
        C_q1,
        C_q2,
        g,
        Jr,
        Lr,
        Mp,
        Lp,
    )

    q1 = (q1 * w_rescale[0]).detach().to(device)
    p1 = (p1 * w_rescale[1]).detach().to(device)
    q2 = (q2 * w_rescale[2]).detach().to(device)
    p2 = (p2 * w_rescale[3]).detach().to(device)

    if min_max_rescale:
        if rescale_dims[0]:
            q1 = (q1 - q1.amin(dim=(1)).unsqueeze(dim=1)) / (
                (q1.amax(dim=(1)) - q1.amin(dim=(1))).abs().unsqueeze(dim=1)
            )
        if rescale_dims[1]:
            q2 = (q2 - q2.amin(dim=(1)).unsqueeze(dim=1)) / (
                (q2.amax(dim=(1)) - q2.amin(dim=(1))).abs().unsqueeze(dim=1)
            )
        if rescale_dims[2]:
            p1 = (p1 - p1.amin(dim=(1)).unsqueeze(dim=1)) / (
                (p1.amax(dim=(1)) - p1.amin(dim=(1))).abs().unsqueeze(dim=1)
            )
        if rescale_dims[3]:
            p2 = (p2 - p2.amin(dim=(1)).unsqueeze(dim=1)) / (
                (p2.amax(dim=(1)) - p2.amin(dim=(1))).abs().unsqueeze(dim=1)
            )

    energy = energy.detach().to(device)
    derivatives = derivatives.detach().to(device)
    t_eval = t_eval.detach().to(device)

    # dataloader to load data in batches
    train_loader, test_loader = data_loader_furuta(
        q1,
        p1,
        q2,
        p2,
        energy,
        derivatives,
        t_eval,
        batch_size=batch_size,
        shuffle=shuffle,
        proportion=proportion,
        coord_type=coord_type,
    )  # u, G,
    return train_loader, test_loader
