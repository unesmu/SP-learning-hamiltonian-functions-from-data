import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
import torch

from torchdiffeq import odeint_adjoint as odeint_adjoint 
# func must be a nn.Module when using the adjoint method
from torchdiffeq import odeint as odeint

# import time as time

from .dynamics import *

''' TRAJECTORIES '''

def random_uniform_two_interval():
    '''
      return x taken from a uniform distribution in one of two intervals 
      [-2,-0.3] or [0.3,2]
    Input : 
      None
    Outputs

    '''
    if torch.tensor([0.5]).bernoulli().bool():
        x = torch.zeros(1).uniform_(0.3,2)
    else:
        x = torch.zeros(1).uniform_(-2,-0.3)
    return x


def get_init_state():
    '''
      Returns initial states q1,p1,q2,p2 for the Furuta pendulum 
      The angular velocities ( /generalized momenmtums) are set to zero
    Inputs : 
      None
    Outputs : 
      y0(tensor) : inital condition
      
    '''
    y0 = torch.zeros(4)

    y0[0] = random_uniform_two_interval() 
    y0[2] = random_uniform_two_interval()

    y0[1]=0.0 # p1 = 0
    y0[3]=0.0 # p2 = 0
    return y0

def get_trajectory_furuta(timesteps=20, y0=None, noise_std=0.0, 
                   u=0.0, C_q1=0.0, C_q2=0.0):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''

    Ts = 0.005 # sampling time 
    t_eval = torch.linspace(1, timesteps, timesteps) * Ts # evaluated times vector
    t_span = [Ts, timesteps*Ts] # [t_start, t_end]

    # get initial state
    if y0 is None:
        y0 = get_init_state()

    # solve the differential equation using odeint
    q_p = odeint(func=lambda t, coords: dynamics_fn_furuta(t, coords, u, C_q1, C_q2), 
                 y0=y0, t=t_eval, method='rk4', options=dict(step_size=Ts)) 

    q1, p1, q2, p2 = torch.chunk(q_p, 4, 1) 

    # add noise 
    q1 = (q1+torch.randn(q1.shape)*noise_std).detach().squeeze()
    p1 = (p1+torch.randn(p1.shape)*noise_std*torch.max(p1)).detach().squeeze()
    q2 = (q2+torch.randn(q2.shape)*noise_std).detach().squeeze()
    p2 = (p2+torch.randn(p2.shape)*noise_std*torch.max(p2)).detach().squeeze()
    # .detach() because the pytorch computational graph is no longer needed
    # only the value is needed
    # .squeeze() to have a desired dimentionality
    # *torch.max(p) otherwise noise is too big compared to the generalized momentum

    return q1, p1, q2, p2, t_eval.detach()

def multiple_trajectories_furuta(time_steps, num_trajectories, 
                          y0=torch.tensor([1.0,0.0,1.0,0.0]),
                          noise_std=0.0, u=0, C_q1=0.0, C_q2=0.0):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    # the first trajectory
    q1, p1, q2, p2, t_eval = get_trajectory_furuta(timesteps=time_steps, 
                                                   y0=y0, noise_std=noise_std, 
                                                   u=u, C_q1=C_q1, C_q2=C_q2)
    
    energy, derivatives = get_energy_furuta(q1, p1, q2, p2, u=u, C_q1=C_q1, C_q2=C_q2)
    derivatives = derivatives.unsqueeze(dim=0)
    for _ in range(num_trajectories-1):
        # the trajectories 2 to num_trajectories
        q1_n, p1_n, q2_n, p2_n, t_eval_n = get_trajectory_furuta(
                     timesteps=time_steps, y0=y0, noise_std=noise_std, u=u, C_q1=0.0, C_q2=0.0)
        energy_n, derivatives_n = get_energy_furuta(q1_n, p1_n, q2_n, p2_n, u=u, C_q1=C_q1, C_q2=C_q2)

        q1 = torch.vstack((q1, q1_n))
        p1 = torch.vstack((p1, p1_n))
        q2 = torch.vstack((q2, q2_n))
        p2 = torch.vstack((p2, p2_n))

        
        energy = torch.vstack((energy, energy_n))
        derivatives  = torch.vstack((derivatives, derivatives_n.unsqueeze(dim=0)))
        # t_eval = torch.vstack((t_eval, t_eval_n))
      
    if num_trajectories == 1 :
        # so that the vectors will have the correct dimensions if only 1
        # trajectory is requested
        q1 = q1.unsqueeze(dim=0)
        p1 = p1.unsqueeze(dim=0)
        q2 = q2.unsqueeze(dim=0)
        p2 = p2.unsqueeze(dim=0)

        energy = energy.unsqueeze(dim=0)
        derivatives = derivatives

        #t_eval = t_eval.unsqueeze(dim=0)
    return q1, p1, q2, p2, energy, derivatives, t_eval


''' ENERGY '''
def energy_furuta(dq1dt,dq2dt,q1):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    # system constants
    g = 9.81
    Jr = 5.72*1e-5
    Lr = 0.085
    Mp = 0.024
    Lp = 0.129
    Jp = (1/12)*Mp*Lp**2

    # function constants
    C1 = Jr + Mp*Lr**2
    C2 = (1/4)*Mp*Lp**2
    C3 = (-1/2)*Mp*Lp*Lr
    C4 = Jp+C2
    C5 = (1/2)*Mp*g*Lp
    
    E = (1/2)* dq2dt**2 * (C1+C2*torch.sin(q1)**2)+ dq2dt*dq1dt*C3*torch.cos(q1)
    E = E +(1/2)*dq1dt**2 *C4+C5*torch.cos(q1)+C5

    return E

def get_energy_furuta(q1, p1, q2, p2, u=0, C_q1=0.0, C_q2=0.0):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    energy=[]
    derivatives=[]
    for coords in torch.stack((q1, p1, q2, p2),dim=1):

      dq1dt, dp1dt, dq2dt, dp2dt = coord_derivatives_furuta(coords, u,
                                                            C_q1, C_q2)

      q1_n = coords[0]
      energy.append(energy_furuta(dq1dt,dq2dt,q1_n))
      derivatives.append(torch.hstack((dq1dt, dp1dt, dq2dt, dp2dt)))
 
    energy = torch.hstack(energy).detach()
    derivatives = torch.vstack(derivatives).detach()
    return energy, derivatives