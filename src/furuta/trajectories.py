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

def random_uniform_two_interval(init_method):
    '''
      return x taken from a uniform distribution in one of two intervals 
      [-2,-0.3] or [0.3,2]
    Input : 
      None
    Outputs

    '''
    if init_method == 'random_nozero':
        if torch.tensor([0.5]).bernoulli().bool():
            x = torch.zeros(1).uniform_(0.3, 2)
        else:
            x = torch.zeros(1).uniform_(-2,-0.3)
    elif init_method == 'random_closetozero':
            x = torch.zeros(1).uniform_(-0.3, 0.3)
    elif init_method == 'random_closetopi':
            x = torch.zeros(1).uniform_(torch.pi-0.3, torch.pi+0.3)
    return x


def get_init_state(init_method):
    '''
      Returns initial states q1,p1,q2,p2 for the Furuta pendulum 
      The angular velocities ( /generalized momenmtums) are set to zero
    Inputs : 
      None
    Outputs : 
      y0(tensor) : inital condition
      
    '''
    y0 = torch.zeros(4)

    y0[0] = random_uniform_two_interval(init_method) 
    y0[2] = torch.zeros(1).uniform_(2, 6) # random_uniform_two_interval() 

    y0[1]=0.0 # p1 = 0
    y0[3]=0.0 # p2 = 0
    return y0

''' Input functions '''

def chirp_fun(t,T=1.5,f0=1,f1=50):
    # https://en.wikipedia.org/wiki/Chirp
    c = (f1-f0)/T
    scale = 1
    return torch.sin(2*torch.pi*(c*t**2/2 + f0*t))*scale

def multi_sine(t):
  scale = 0.5
  f = torch.tensor([2,10,3,4]).unsqueeze(dim=1)
  A = torch.tensor([2,0.5,0.3,0.8]).unsqueeze(dim=1)
  return (A*torch.sin(2*torch.pi*t*f)).sum(dim=0)*scale

def step_fun(t, t1=0.05, value=0.1):
  f = torch.zeros_like(t)
  f[t < t1] = 0
  f[~(t < t1)] = value
  return f

def u_func(t, utype):  
  ''' time dependent input '''
  if utype == 'tanh':
      u = (-torch.tanh((t-0.75)*4)+1)/100
  elif utype == 'CHIRP':
      u = chirp_fun(t,T=1.5,f0=1,f1=10)
  elif utype == 'multisine':
      u = multi_sine(t)
  elif utype == 'step':
      u = step_fun(t, t1=0.5)
  u.requires_grad=False
  return u

def g_func(coords, gtype):
  ''' state dependent input '''
  # q1,p1,q2,p2 = torch.split(coords,1)
  if gtype=='simple':
      g = torch.tensor([0,0,1,1]) 
  g.requires_grad=False
  return g

'''Functions to generate the trajectories'''

def get_trajectory_furuta(utype, gtype, init_method, u_func=None, g_func=None, time_steps=20, y0=None, noise_std=0.0, Ts = 0.005, C_q1=0.0, 
                          C_q2=0.0, g = 9.81, Jr = 5.72*1e-5, Lr = 0.085, Mp = 0.024, Lp = 0.129):   
    '''
    Description:

    Inputs:

    Outputs:
      
    '''

     # sampling time 
    t_eval = torch.linspace(1, time_steps, time_steps) * Ts # evaluated times vector
    t_span = [Ts, time_steps*Ts] # [t_start, t_end]

    # get initial state
    if y0 is None:
        y0 = get_init_state(init_method)

    # solve the differential equation using odeint
    q_p = odeint(func=lambda t, coords: dynamics_fn_furuta(t, coords, C_q1, C_q2, g, Jr, Lr, Mp, Lp, u_func, g_func, utype, gtype), 
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


''' ENERGY functions '''

def energy_furuta(dq1dt, dq2dt, q1, g, Jr, Lr, Mp, Lp):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    # system constants
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

def get_energy_furuta(time_steps, Ts, u_func, g_func, utype, gtype, q1, p1, q2, p2, C_q1, C_q2, g, Jr, Lr, Mp, Lp):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    energy=[]
    derivatives=[]
    t = torch.linspace(1, time_steps, time_steps) * Ts
    #for t, coords in (t, torch.stack((q1, p1, q2, p2),dim=1)):
    coords = torch.stack((q1, p1, q2, p2),dim=1)
    for i in range(len(t)):

      dq1dt, dp1dt, dq2dt, dp2dt = coord_derivatives_furuta(t[i], coords[i,:],
                                                            C_q1, C_q2, g, Jr, Lr, Mp, Lp,u_func, g_func, utype, gtype)

      q1_n = coords[i,0]
      energy.append(energy_furuta(dq1dt,dq2dt,q1_n, g, Jr, Lr, Mp, Lp))
      derivatives.append(torch.hstack((dq1dt, dp1dt, dq2dt, dp2dt)))
 
    energy = torch.hstack(energy).detach()
    derivatives = torch.vstack(derivatives).detach()
    return energy, derivatives

def get_energy_furuta_newtonian(q1, dq1dt, q2, dq2dt, C_q1, C_q2, g, Jr, Lr, Mp, Lp):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    energy=[]
    derivatives=[]
    for coords in torch.stack((q1, dq1dt, dq2dt),dim=1):

      q1_n = coords[0]
      dq1dt = coords[1]
      dq2dt = coords[2]
      energy.append(energy_furuta(dq1dt,dq2dt,q1_n, g, Jr, Lr, Mp, Lp))

    energy = torch.hstack(energy).detach()
    return energy

''' MULTIPLE TRAJECTORIES '''

def multiple_trajectories_furuta(utype, gtype, init_method, time_steps, num_trajectories, u_func=None, g_func=None, 
                          y0=torch.tensor([1.0,0.0,1.0,0.0]), Ts = 0.005,
                          noise_std=0.0, C_q1=0.0, C_q2=0.0, g = 9.81, 
                          Jr = 5.72*1e-5, Lr = 0.085, Mp = 0.024, Lp = 0.129, energ_deriv=True):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    # the first trajectory
    q1, p1, q2, p2, t_eval = get_trajectory_furuta(utype, gtype, init_method, u_func, g_func, time_steps, y0, noise_std, 
                                                   Ts, C_q1, C_q2, g, Jr, Lr, 
                                                   Mp , Lp)
    energy = []
    derivatives = []
    if energ_deriv:
      energy, derivatives = get_energy_furuta(time_steps, Ts, u_func, g_func, utype, gtype, q1, p1, q2, p2, C_q1, C_q2, g, Jr, Lr, Mp, Lp)
      derivatives = derivatives.unsqueeze(dim=0)

    for _ in range(num_trajectories-1):
        # the trajectories 2 to num_trajectories
        q1_n, p1_n, q2_n, p2_n, t_eval_n = get_trajectory_furuta(utype, gtype, init_method, u_func, g_func, time_steps, 
                                                                 y0, noise_std, 
                                                                Ts, C_q1, C_q2, g, Jr, Lr, 
                                                                Mp , Lp)
        energy_n, derivatives_n = get_energy_furuta(time_steps, Ts, u_func, g_func, utype, gtype, q1_n, p1_n, q2_n, p2_n, C_q1, C_q2, g, Jr, Lr, Mp, Lp)

        q1 = torch.vstack((q1, q1_n))
        p1 = torch.vstack((p1, p1_n))
        q2 = torch.vstack((q2, q2_n))
        p2 = torch.vstack((p2, p2_n))

        if energ_deriv:
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
        if energ_deriv:
          energy = energy.unsqueeze(dim=0)
          derivatives = derivatives

        # t_eval = t_eval.unsqueeze(dim=0)

    return q1, p1, q2, p2, energy, derivatives, t_eval