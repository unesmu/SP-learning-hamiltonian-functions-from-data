import torch
from .dynamics import *
from torchdiffeq import odeint as odeint

# Similar to https://github.com/Physics-aware-AI/Symplectic-ODENet/blob/master/experiment-single-force/data.py


def chirp_fun(t, T=1.5, f0=1, f1=50, scale=1):
    """
    chirp function implemented using the formula from
    https://en.wikipedia.org/wiki/Chirp
    """

    c = (f1 - f0) / T
    return torch.sin(2 * torch.pi * (c * t**2 / 2 + f0 * t)) * scale


def multi_sine(t, scale=0.5):
    """
    Multi-sine function, implemented as the sum of multiple sine functions
    """
    f = torch.tensor([2, 10, 3, 4], device=t.device).unsqueeze(dim=1)
    A = torch.tensor([2, 0.5, 0.3, 0.8], device=t.device).unsqueeze(dim=1)
    return (A * torch.sin(2 * torch.pi * t * f)).sum(dim=0) * scale


def step_fun(t, t1=0.05, scale=0.1):
    """
    Simple step function
    """
    f = torch.zeros_like(t)
    f[t < t1] = 0
    f[~(t < t1)] = scale
    return f


def sine_fun(t, scale=0.5, f=1):
    return scale * torch.sin(2 * torch.pi * t * f)


class U_FUNC:

    """
        Class that contains the input functions
        Example use :
            utype = 'chirp's
            u_func = U_FUNC(utype=utype)
            u_func.params['T'] = 2.0
            u_func.params['f0'] = 0
            u_func.params['f1'] = 1
            u_func.params['scale'] = 1

    """

    def __init__(self, utype=None, params={}):
        super(U_FUNC).__init__()
        self.utype = utype
        self.params = params  # dict containing params for the input function
        self.params["T"] = 2.0
        self.params["f0"] = 0
        self.params["f1"] = 1
        self.params["scale"] = 1

    def forward(self, t):
        """time dependent input"""
        if self.utype == "tanh":
            u = (-torch.tanh((t - 0.75) * 4) + 1) / 100
        elif self.utype == "chirp":
            u = chirp_fun(t, T=self.params["T"], f0=self.params["f0"], f1=self.params["f1"], scale=self.params["scale"])
        elif self.utype == "multisine":
            u = multi_sine(t, scale=self.params["scale"])
        elif self.utype == "step":
            u = step_fun(t, t1=0.5)
        elif self.utype == "sine":
            u = sine_fun(t, scale=self.params["scale"], f=self.params["f1"])
        elif self.utype is None:
            u = torch.zeros(t.shape, device=t.device)
        u.requires_grad = False
        return u


class G_FUNC:
    """
    Class that contains the input matrix functionss
    """

    def __init__(self, device, gtype=None, params={}):
        super(G_FUNC).__init__()
        self.gtype = gtype
        self.params = params  # dict containing params on
        self.params["q_ref"] = torch.tensor([1.0], device=device)

    def forward(self, coords):
        """state dependent input"""
        q, p = torch.split(coords, 1, dim=-1)
        if self.gtype == "simple":
            if len(q.shape) == 1:
                dimension = 0
            else:
                dimension = 1
            g = torch.stack((torch.zeros(q.shape[0]), torch.ones(q.shape[0])), dim=dimension)

        elif self.gtype is None:
            if len(q.shape) == 1:
                dimension = 0
            else:
                dimension = 1
            g = torch.stack((torch.zeros(q.shape[0]), torch.zeros(q.shape[0])), dim=dimension)
        g.requires_grad = False
        return g


def get_trajectory_pend(device, time_steps, Ts, y0, noise_std, C, m, g, l, u_func, g_func):
    """
    Similar to SymODEN repository
    """
    # evaluated times vector
    t_eval = torch.linspace(1, time_steps, time_steps, device=device) * Ts

    # get initial state
    if y0 is None:
        y0 = torch.rand(2) * 4.0 - 2  # uniform law [-2,2]
        y0[1] = 0

    q_p = odeint(
        func=lambda t, coords: dynamics_fn_pend(t, coords, C, m, g, l, u_func, g_func),
        y0=y0,
        t=t_eval,
        method="rk4",
        options=dict(step_size=Ts),
    )

    q, p = torch.chunk(q_p, 2, 1)

    # add noise
    q = q + torch.randn(q.shape, device=device) * noise_std
    p = p + torch.randn(p.shape, device=device) * noise_std

    return q.detach().squeeze(), p.detach().squeeze(), t_eval.detach()


def energy_pendulum(theta_dot, theta, m, g, l):
    """
    This function calculates the energy of the simple pendulum in function of
    the angle theta and it's derivative
    Inputs :
        - theta_dot :
        - theta :
        - m : pendulum mass
        - g :
        - l : length of arm
    Outputs:
        - E () : Energy of the system
    """
    E = (1 / 2) * m * (l**2) * theta_dot**2 + m * g * l * (1 - torch.cos(theta))
    return E


def get_energy_pendulum(t_eval, u_func, g_func, q, p, C, m, g, l):
    """
    Returns the energy at each time step given the time vector t_eval
    """
    energy = []
    derivatives = []

    coords = torch.stack((q, p), dim=1)
    for i in range(len(t_eval)):

        dqdt, dpdt = coord_derivative_pend(t_eval[i], coords[i, :], C, m, g, l, u_func, g_func)

        theta_dot = dqdt
        theta = coords[i, 0]

        energy.append(energy_pendulum(theta_dot, theta, m, g, l))
        derivatives.append(torch.hstack((dqdt, dpdt)))

    energy = torch.hstack(energy).detach()
    derivatives = torch.vstack(derivatives).detach()
    return energy.squeeze(), derivatives.squeeze()


def multiple_trajectories(
    time_steps, num_trajectories, device, Ts, y0, noise_std, C, m, g, l, u_func, g_func, coord_type="hamiltonian"
):
    """
    Generates the trajectories (all generalized coordinates and energy)
    """

    # the first trajectory
    q, p, t_eval = get_trajectory_pend(device, time_steps, Ts, y0, noise_std, C, m, g, l, u_func, g_func)
    energy, derivatives = get_energy_pendulum(t_eval, u_func, g_func, q, p, C, m, g, l)
    energy = energy.squeeze()
    q = q.squeeze()
    if coord_type == "hamiltonian":
        p = p.squeeze()
    if coord_type == "newtonian":
        p = derivatives[:, 0].squeeze()

    derivatives = derivatives.unsqueeze(dim=0)

    for _ in range(num_trajectories - 1):
        # the trajectories 2 to num_trajectories
        q_n, p_n, _ = get_trajectory_pend(device, time_steps, Ts, y0, noise_std, C, m, g, l, u_func, g_func)
        energy_n, derivatives_n = get_energy_pendulum(t_eval, u_func, g_func, q_n, p_n, C, m, g, l)

        energy = torch.vstack((energy, energy_n.squeeze()))
        q = torch.vstack((q, q_n.squeeze()))

        if coord_type == "hamiltonian":
            p = torch.vstack((p, p_n.squeeze()))
            p = p.squeeze()
        if coord_type == "newtonian":
            p = torch.vstack((p, derivatives_n[:, 0].squeeze()))

        derivatives = torch.vstack((derivatives, derivatives_n.unsqueeze(dim=0)))

    if num_trajectories == 1:
        # so that the vectors will have the correct dimensions if only 1
        # trajectory is requested
        q = q.unsqueeze(dim=0)
        p = p.unsqueeze(dim=0)
        energy = energy.unsqueeze(dim=0)
    return q.t(), p.t(), t_eval, energy.t(), torch.permute(derivatives, (1, 0, 2))
