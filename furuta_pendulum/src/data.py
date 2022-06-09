from torch.utils.data import Dataset, DataLoader, random_split
import torch

from torchdiffeq import odeint_adjoint as odeint_adjoint
# func must be a nn.Module when using the adjoint method
from torchdiffeq import odeint as odeint


class TrajectoryDataset_furuta(Dataset):
    '''
    Description:

    Inputs:

    Outpus:

    '''

    def __init__(self, q1, p1, q2, p2, t_eval,
                 derivatives, coord_type='hamiltonian'):
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

        if self.coord_type == 'hamiltonian':

            q1 = self.q1[idx]
            p1 = self.p1[idx]
            q2 = self.q2[idx]
            p2 = self.p2[idx]

            x = torch.stack((q1, p1, q2, p2), dim=1)

        if self.coord_type == 'newtonian':

            q1 = self.q1[idx]
            q2 = self.q2[idx]
            dq1dt = self.dq1dt[idx]
            dq2dt = self.dq2dt[idx]

            x = torch.stack((q1, dq1dt, q2, dq2dt), dim=1)

        t_eval = self.t_eval

        return x, t_eval


def data_loader_furuta(q1, p1, q2, p2, energy, derivatives, t_eval, batch_size,
                       shuffle=True, proportion=0.5, coord_type='hamiltonian'):
    '''
    Description:

    Inputs:

    Outpus:

    '''
    # split  into train and test
    full_dataset = TrajectoryDataset_furuta(
        q1, p1, q2, p2, t_eval, derivatives, coord_type=coord_type)
    if proportion:

        train_size = int(proportion * len(full_dataset))
        test_size = len(full_dataset) - train_size

        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size])

        test_loader = DataLoader(
            test_dataset,
            batch_size,
            shuffle
        )
    else:
        # if proportion is set to None don't split the dataset
        train_dataset = full_dataset
        test_loader = None

    # create the dataloader object from the custom dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle
    )

    return train_loader, test_loader
