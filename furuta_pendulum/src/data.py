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

    def __init__(self, q1, p1, q2, p2, t_eval,  # energy=torch.tensor(False),
                 derivatives, coord_type='hamiltonian'):  # u, G,
        self.t_eval = t_eval
        self.coord_type = coord_type
        self.q1 = q1  # [num_trajectories, time_steps]
        self.p1 = p1  # [num_trajectories, time_steps]
        self.q2 = q2  # [num_trajectories, time_steps]
        self.p2 = p2  # [num_trajectories, time_steps]
        # self.u = u # [num_trajectories, time_steps]
        # self.G = G # [num_trajectories, time_steps, (g1,g2,g3,g4)]
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
            # u = self.u[idx]
            # G = self.G[idx]
            x = torch.stack((q1, p1, q2, p2), dim=1)  # ,u
            # x = torch.hstack((x,G))

        if self.coord_type == 'newtonian':

            q1 = self.q1[idx]
            q2 = self.q2[idx]
            dq1dt = self.dq1dt[idx]
            dq2dt = self.dq2dt[idx]

            x = torch.vstack((q1, dq1dt, q2, dq2dt))

        t_eval = self.t_eval

        return x, t_eval


def data_loader_furuta(q1, p1, q2, p2, energy, derivatives, t_eval, batch_size,
                       shuffle=True, proportion=0.5, coord_type='hamiltonian'):  # u, G,
    '''
    Description:

    Inputs:

    Outpus:

    '''
    # split  into train and test
    full_dataset = TrajectoryDataset_furuta(
        q1, p1, q2, p2, t_eval, derivatives, coord_type=coord_type)  # u, G,
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
        energy_train = energy
        derivatives_train = derivatives
        t_eval_train = t_eval
        test_loader = None

    # create the dataloader object from the custom dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle
    )

    return train_loader, test_loader
