import torch

def pendulum_H(q, p, m, g, l):
    H = (1/(2*m*(l**2)))*p**2 + m*g*l*(1-torch.cos(q))  
    return H

def hamiltonian_fn_pend(coords, m, g, l):
    q, p = torch.split(coords,1)
    H = pendulum_H(q,p, m, g, l)
    return H

def coord_derivative_pend(t, coords, C, m, g, l, u_func, g_func):

    if coords.requires_grad is not True:
      coords.requires_grad=True
    H = hamiltonian_fn_pend(coords, m, g, l)

    # derivaties of the hamiltornian function wrt the generalized coordinates
    dcoords = torch.autograd.grad(H, coords, create_graph=True)
    
    dHdq, dHdp = torch.split(dcoords[0],1)

    u = u_func.forward(t)
    G = g_func.forward(coords)

    dqdt = dHdp + G[0]*u
    dpdt = -dHdq - C*dHdp + G[1]*u

    return dqdt, dpdt

def dynamics_fn_pend(t, coords, C, m, g, l, u_func, g_func):

    dqdt, dpdt = coord_derivative_pend(t, coords, C, m, g, l, u_func, g_func)
    # symplectic gradient
    S = torch.hstack((dqdt, dpdt)) 
    return S 