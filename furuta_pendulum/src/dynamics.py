import torch
from torchdiffeq import odeint as odeint


def furuta_H(q1, p1, q2, p2, g, Jr, Lr, Mp, Lp):
    """
    Description:
        Hamiltonian function for the Furuta pendulum

    Inputs:
        - q1,p1,q2,p2 (tensors): Generalized coordinates
        - g (floats): gravitational acceleration
        - Jr (floats): Inertia of rotary arm 2
        - Lr (floats): Length of rotary arm 2 (moves with coordinate q1 or alpha in report)
        - Mp (floats): mass af the weight attached at the edge of arm 2
        - Lp (floats): Length of rotary arm 1 (moves with coordinate q2 or theta in report)
    Outputs:
        - H (tensor): Hamiltonian function

    Credits:
        - Equations & constants are from Jonas's report see reference [7] from my report
    """

    # system constants
    Jp = (1 / 12) * Mp * Lp**2

    # function constants
    C1 = Jr + Mp * Lr**2
    C2 = (1 / 4) * Mp * Lp**2
    C3 = (-1 / 2) * Mp * Lp * Lr
    C4 = Jp + C2
    C5 = (1 / 2) * Mp * g * Lp

    # hamiltonian function
    H = (
        p1**2 * (C1 + C2 * torch.sin(q1) ** 2)
        + C4 * p2**2
        - 2 * p1 * p2 * C3 * torch.cos(q1)
    )
    H = (
        (1 / 2)
        * (H)
        / (C1 * C4 + C4 * C2 * torch.sin(q1) ** 2 - (C3**2) * (torch.cos(q1) ** 2))
    )
    H = H + C5 * (torch.cos(q1) + 1)

    return H


def hamiltonian_fn_furuta(coords, g, Jr, Lr, Mp, Lp):
    """
    Description:
        Hamiltonian function for the Furuta pendulum, wraps furuta_H() so that it is in
        the right format for ODEint

    Inputs:
        - coords (tensor): vector containing generalized coordinates q1,p1,q2,p2
        - g, Jr, Lr, Mp, Lp (floats): furuta pendulum parameters (same as in furuta_H())
      Outputs:
        - H (tensor): Scalar Hamiltonian function

    Credits:
        This function takes the same structure as the one in the SymODEN repository
    """
    q1, p1, q2, p2 = torch.chunk(coords, 4, dim=-1)

    H = furuta_H(q1, p1, q2, p2, g, Jr, Lr, Mp, Lp)

    return H


def coord_derivatives_furuta(t, coords, C_q1, C_q2, g, Jr, Lr, Mp, Lp, u_func, g_func):
    """
    Description:
        Returns the derivatives of the generalized coordinates with respect
        to (w.r.t) time

    Inputs:
        - coords (tensor): vector containing generalized coordinates q1,p1,q2,p2
        - C_q1 (float): coefficient of friction related to p1 (and q1)
        - C_q2 (float): coefficient of friction related to p2 (and q2)
        - g, Jr, Lr, Mp, Lp (floats): furuta pendulum parameters

    Outputs:
        - dq1dt, dp1dt, dq2dt, dp2dt (tensors): Derivatives w.r.t coords
    """
    if coords.requires_grad is not True:
        coords.requires_grad = True

    # Hamiltonian function
    H = hamiltonian_fn_furuta(coords, g, Jr, Lr, Mp, Lp)

    # gradient of the hamiltornian function wrt the generalized coordinates
    dcoords = torch.autograd.grad(H.sum(), coords, create_graph=True)
    dHdq1, dHdp1, dHdq2, dHdp2 = torch.chunk(dcoords[0], 4, dim=-1)

    # evaluate input scalar u and matrix G
    U = u_func.forward(t)
    G = g_func.forward(coords)

    # symplectic gradient / derivatives of the generalized coordinates w.r.t time
    dq1dt = dHdp1 + (U * G[:, 0]).unsqueeze(dim=1)
    dp1dt = -dHdq1 - C_q1 * dHdp1 + (U * G[:, 1]).unsqueeze(dim=1)
    dq2dt = dHdp2 + (U * G[:, 2]).unsqueeze(dim=1)
    dp2dt = -dHdq2 - C_q2 * dHdp2 + (U * G[:, 3]).unsqueeze(dim=1)

    return dq1dt, dp1dt, dq2dt, dp2dt


def dynamics_fn_furuta(t, coords, C_q1, C_q2, g, Jr, Lr, Mp, Lp, u_func, g_func):
    """
    Description:
        Function that returns the gradient (in form of a function) of a Hamiltonian function

    Inputs:
        - t (tensor): time at which the derivaties must be evaluated
        - coords (tensor): vector containing generalized coordinates q1,p1,q2,p2
        - C_q1 (float): coefficient of friction related to p1 (and q1)
        - C_q2 (float): coefficient of friction related to p2 (and q2)
        - g, Jr, Lr, Mp, Lp (floats): furuta pendulum parameters
        - u_func (function): input scalar function
        - g_func (function): input matrix function

    Outputs:
        - S (tensor): Symplectic gradient / derivatives of the generalized coordinates w.r.t time

    Credits:
        - This function has a similar structure as the one in the SymODEN repository
    """

    dq1dt, dp1dt, dq2dt, dp2dt = coord_derivatives_furuta(
        t, coords, C_q1, C_q2, g, Jr, Lr, Mp, Lp, u_func, g_func
    )

    S = torch.hstack((dq1dt, dp1dt, dq2dt, dp2dt))
    return S


""" Input functions """


def chirp_fun(t, T=1.5, f0=1, f1=50, scale=1):
    """
    Description:
        chirp function evaluated at t, using the formula from
        https://en.wikipedia.org/wiki/Chirp
    Inputs:
        - t (tensor): time steps at which it is evaluated
        - scale (Float): multiplier to change the magnitude of input
        - f1 (int) : Frequency at time T
        - f0 (int) : Frequency at time 0
        - T (float) : Time at which we have f1
    Outputs:
        - (tensor) : chirp function evaluated at the timesteps t

    """
    # 
    c = (f1 - f0) / T
    return torch.sin(2 * torch.pi * (c * t**2 / 2 + f0 * t)) * scale


def multi_sine(t, scale=0.5):
    """
    Description:
        Multi sine input function (Sum of sines)
    Inputs:
        - t (tensor): time steps at which it is evaluated
        - scale (Float): multiplier to change the magnitude of input
    Outputs:
        - (tensor) : multisine function evaluated at the timesteps t
    """
    f = torch.tensor([2, 10, 3, 4], device=t.device).unsqueeze(dim=1)
    A = torch.tensor([2, 0.5, 0.3, 0.8], device=t.device).unsqueeze(dim=1)
    return (A * torch.sin(2 * torch.pi * t * f)).sum(dim=0) * scale


def sine_fun(t, scale=0.5, f=1):
    """
    Description:
        Sine function evaluated at t
    Inputs:
        - t (tensor): time steps at which it is evaluated
        - scale (Float): multiplier to change the magnitude of input
        - f (int) : Frequency
    Outputs:
        - (tensor) : sine function evaluated at the timesteps t

    """
    return scale * torch.sin(2 * torch.pi * t * f)


def step_fun(t, t1=0.05, scale=0.1):
    """
    Description:
        step function evaluated at t
    Inputs:
        - t (tensor): time steps at which it is evaluated
        - scale (Float): multiplier to change the magnitude of input
        - t1 (int) : where the step happens
    Outputs:
        - (tensor) : step function evaluated at the timesteps t

    """
    f = torch.zeros_like(t)
    f[t < t1] = 0
    f[~(t < t1)] = scale
    return f


class U_FUNC:
    """
    Description:
        Class that instantiates the input functino
    Inputs:
        - utype (string): can be one of :
                                    chirp
                                    sine
                                    tanh
                                    multisine
                                    step
    Methods:
        - forward(self, t) : use time t to evalute the chosen
                            input function at time t

    Example use : 
    input_function = U_FUNC(utype='chirp')
    evaluation = input_function.forward()

    """

    def __init__(self, utype=None, params={}):
        super(U_FUNC).__init__()
        self.utype = utype
        self.params = params  # dict containing parameters for the input function
        self.params["T"] = 1.5
        self.params["f0"] = 1
        self.params["f1"] = 10
        self.params["scale"] = 1

    def forward(self, t):
        """time dependent input"""
        if self.utype == "chirp":
            u = chirp_fun(
                t,
                T=self.params["T"],
                f0=self.params["f0"],
                f1=self.params["f1"],
                scale=self.params["scale"],
            )
        elif self.utype == "sine":
            u = sine_fun(t, scale=self.params["scale"], f=self.params["f1"])
        elif self.utype == "tanh":
            u = (-torch.tanh((t - 0.75) * 4) + 1) / 100
        elif self.utype == "multisine":
            u = multi_sine(t, scale=self.params["scale"])
        elif self.utype == "step":
            u = step_fun(t, t1=0.5)
        elif self.utype is None:
            u = torch.zeros(t.shape, device=t.device)
        u.requires_grad = False
        return u


class G_FUNC:
    """
    Description:

    Inputs:
        - ():

    Outputs:
        - ():
    """

    def __init__(self, gtype=None, params={}):
        super(G_FUNC).__init__()
        self.gtype = gtype
        self.params = params  # dict containing parameters

    def forward(self, coords):
        """state dependent input"""
        q1, p1, q2, p2 = torch.split(coords, 1, dim=-1)
        if self.gtype == "simple":
            if len(q1.shape) == 1:
                dimension = 0
            else:
                dimension = 1
            g = torch.stack(
                (
                    torch.zeros(q1.shape[0], device=q1.device),  # (q1,p1,q2,p2)
                    torch.ones(q1.shape[0], device=q1.device),
                    torch.zeros(q1.shape[0], device=q1.device),
                    torch.ones(q1.shape[0], device=q1.device),
                ),
                dim=dimension,
            )

        elif self.gtype is None:
            if len(q1.shape) == 1:
                dimension = 0
            else:
                dimension = 1
            g = torch.stack(
                (
                    torch.zeros(q1.shape[0], device=q1.device),  # (q1,p1,q2,p2)
                    torch.zeros(q1.shape[0], device=q1.device),
                    torch.zeros(q1.shape[0], device=q1.device),
                    torch.zeros(q1.shape[0], device=q1.device),
                ),
                dim=dimension,
            )
        g.requires_grad = False
        return g
