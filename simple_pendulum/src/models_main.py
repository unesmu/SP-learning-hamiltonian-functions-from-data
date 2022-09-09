import torch

""" SIMPLE  HNN """


class Simple_HNN(torch.nn.Module):
    """
    Modified version of the original SymODEN_R module
    Equivalent of unconstrained ODE HNN from the report
    Architecture for input (q, p, u),
    where q and p are tensors of size (bs, n) and u is a tensor of size (bs, 1)
    """

    def __init__(self, H_net=None, device=None, dissip=False):
        super(Simple_HNN, self).__init__()

        self.H_net = H_net

        self.device = device

        # add a learnable dissipation coefficient
        self.dissip = dissip
        self.C_dissip = torch.nn.Parameter(torch.tensor([0.5]))  # torch.nn.Parameter(torch.rand(1))
        self.C_dissip.requires_grad = True

    def forward(self, t, x):
        with torch.enable_grad():
            q_p = x

            q, p = torch.chunk(x, 2, dim=-1)

            q_p.requires_grad_(True)

            H = self.H_net(q_p)

            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)

            dH = dH[0]

            dHdq, dHdp = torch.chunk(dH, 2, dim=-1)

            dqdt = dHdp
            if self.dissip:
                dpdt = -dHdq - self.C_dissip * dHdp
            else:
                dpdt = -dHdq

            # symplectic gradient
            S_h = torch.cat((dqdt, dpdt), dim=-1)
            return S_h


""" INPUT HNN """


class Input_HNN(torch.nn.Module):
    """
    Modified version of the original SymODEN_R module from symoden repository
    Similar to unconstrained ODE HNN from the report

    """

    def __init__(self, u_func=None, G_net=None, H_net=None, device=None, dissip=False):
        super(Input_HNN, self).__init__()
        self.H_net = H_net
        self.G_net = G_net
        self.u_func = u_func

        self.device = device
        self.dissip = dissip
        # add learnable dissipation coefficients
        self.C = torch.nn.Parameter(
            torch.tensor([0.02]).sqrt()
        )  # torch.nn.Parameter(torch.randn(1)+1) # torch.nn.Parameter(torch.tensor([0.5]))
        self.C.requires_grad = True

    def forward(self, t, x):
        with torch.enable_grad():
            q_p = x

            # q1,p1,q2,p2 = torch.chunk(x,4,dim=-1)

            q_p.requires_grad_(True)

            H = self.H_net(q_p)

            # .sum() to sum up the hamiltonian funcs of a batch
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)

            dH = dH[0]

            dHdq, dHdp = torch.chunk(dH, 2, dim=-1)

            G = self.G_net.forward(q_p)

            if self.u_func:
                u = self.u_func.forward(t)
            else:
                u = torch.tensor([0.0])
            dqdt = dHdp + (G[:, 0].T * u).unsqueeze(dim=1)
            if self.dissip:
                dpdt = -dHdq + (G[:, 1].T * u).unsqueeze(dim=1) - self.C.pow(2) * dHdp
            else:
                dpdt = -dHdq + (G[:, 1].T * u).unsqueeze(dim=1)
            # symplectic gradient
            S_h = torch.cat((dqdt, dpdt), dim=-1)

            return S_h

    def freeze_G_net(self, freeze=True):
        """
        Only freez the G_net parameters
        Inputs
            freeze(bool) : True = freeze the parameters; False = don't

        outputs:
            None

        """
        for param in self.G_net.parameters():
            param.requires_grad = not freeze

    def freeze_H_net(self, freeze=True):
        """
        Only freez the H_net parameters
        Inputs
            freeze(bool) : True = freeze the parameters; False = don't

        outputs:
            None
        """
        for param in self.H_net.parameters():
            param.requires_grad = not freeze
