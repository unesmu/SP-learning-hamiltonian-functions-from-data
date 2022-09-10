import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

from torch.utils.data import Dataset, DataLoader, random_split
import torch

from torchdiffeq import odeint_adjoint as odeint_adjoint
# func must be a nn.Module when using the adjoint method
from torchdiffeq import odeint as odeint

import time as time

import json

from .autoencoder_plots import *
from .trajectories import *
from .dynamics import *
from .train import *

def train_only_ae(autoencoder,
                  device,
                  model,
                  train_loader,
                  test_loader,
                  epochs,
                  horizon,
                  lr=1e-3,
                  w=torch.tensor([0.1, 0.1, 1.0, 1.0])):


    optim = torch.optim.Adam(autoencoder.parameters(), lr, weight_decay=1e-4)

    stats = {'train_loss': [], 'test_loss': []}

    for step in range(epochs):

        train_loss = 0
        test_loss = 0
        ae_loss = 0
        t1 = time.time()

        autoencoder.eval()
        
        for x, t_eval in iter(train_loader): # x is [batch_size,time_steps, (q1,p1,q2,p1)]

            t_eval = t_eval[0, :horizon]

            _, x_hat = autoencoder(x[:, :horizon, :])
            # autoencoder only outputs q_dot_hat states = x_hat

            train_loss_batch = L2_loss(x_hat, x[:, :horizon, :], w)
            train_loss = train_loss + train_loss_batch.item()

            if not step % 50:
                n = 0
                print_ae_train(x_hat, x, n, horizon)

            train_loss_batch.backward()
            optim.step()
            optim.zero_grad()

        t2 = time.time()
        train_time = t2-t1
        if not step % 20:
            print('epoch {:4d} | train time {:.2f} | train loss {:12e} '
                  .format(step, train_time, train_loss))

        # logging
        stats['train_loss'].append(train_loss)

    return stats


def train_ae(model,
             device,
             autoencoder,
             train_loader,
             test_loader,
             Ts,
             horizon=False,
             horizon_type=False,
             horizon_list=[50, 100, 150, 200, 250, 300],
             switch_steps=[200, 200, 200, 150, 150, 150],
             steps_ae=5,
             epoch_number=20,
             w=torch.tensor([0.1, 0.1, 1.0, 1.0])):
    '''
    Description:

    Inputs:

    Outpus:

    '''
    # TODO : make a function for one epoch

    alpha = 1.0
    beta = 1.0
    gamma = 0.5

    lr = 1e-3
    Ts = 0.005
    params = list(model.parameters()) + list(autoencoder.parameters())
    optim = torch.optim.Adam(params, lr, weight_decay=1e-4)

    stats = {'train_loss': [], 'test_loss': []}
    # weight for loss function

    for step in range(epoch_number):

        train_loss = 0
        test_loss = 0
        ae_loss = 0
        t1 = time.time()

        if horizon_type == 'auto':
            _, horizon = select_horizon_list(step, epoch_number, horizon_list, switch_steps)

        # x is [batch_size,time_steps,(q1,p1,q2,p1)]
        for x, t_eval in iter(train_loader):

            t_eval = t_eval[0, :horizon]


            # x_hat is the reconstructed nominal trajectory ; q_dot_hat = decoder(encoder(nominaltrajectory))
            # z is nominal trajectory in latent space (encoded nominal trajectory)
            z, x_hat = autoencoder(x[:, :horizon, :])
            # z is [batch_size,time_steps,(q1,p1,q2,p1)]

            # model output in latent space
            train_z_hat = odeint(
                model, z[:, 0, :], t_eval, method='rk4', options=dict(step_size=Ts))
            # train_z_hat is [time_steps, batch_size, (q1,p1,q2,p1)]

            # decoded output trajectory
            train_x_hat = autoencoder.decoder(train_z_hat[:, :, :])
            # train_x_hat = [time_steps, batch_size, (q1_HNN_decoded, p1_HNN_decoded, q2_HNN_decoded, p1_HNN_decoded)]

            # loss between HNN output and encoded nominal trajectory (nominal trajectory in latent space)
            loss_HNN_batch = L2_loss(z[:, :, :].permute(
                (1, 0, 2)), train_x_hat[:, :, :], w)  # [:,:horizon])

            # loss between decoded HNN output and nominal trajectory
            loss_prediction_batch = L2_loss(
                x[:, :horizon, :],  train_x_hat[:, :, :].permute((1, 0, 2)), w)

            # loss between nominal trajectory and decoded(encoded(nominaltrajectory))
            # to make sure the autoencoder properly reconstructs the nominal trajectory
            loss_AE_batch = L2_loss(
                x[:, :horizon, :], x_hat[:, :horizon, :], w)

            # train_loss_batch = L2_loss(torch.permute(z[:, :, :horizon], (2,0,1)) , train_z_hat[:horizon,:,:],w)#[:,:horizon])
            train_loss_batch = alpha*loss_HNN_batch + beta * \
                loss_prediction_batch + gamma*loss_AE_batch

            if not step % 10:
                n = 0
                fig, ax = plt.subplots(1, 4, figsize=(
                    15, 4), constrained_layout=True, sharex=True)
                list_1 = [r'$q1$', r'$\dot{q1}[rad/s]$',
                          r'$q2$', r'$\dot{q2}[rad/s]$']
                list_2 = ['a', 'b', 'c', 'd']
                for i in range(4):
                    ax[i].plot(t_eval.detach().cpu(), x[n, :horizon,
                               i].detach().cpu(), label='nominal')
                    ax[i].plot(t_eval.detach().cpu(), x_hat[n, :,
                               i].detach().cpu(), '--', label='autencoder')
                    ax[i].plot(t_eval.detach().cpu(), train_x_hat[:,
                               n, i].detach().cpu(), '--', label='HNN_decoded')
                    ax[i].set_title(list_1[i], fontsize=10)
                    ax[i].set_ylabel(list_1[i])
                    ax[i].set_xlabel('time (s)')

                ax[3].legend()
                plt.suptitle(
                    'Autoencoder output compared to nominal and predicted(HNN) trajectories (Newtonian coordinates)')
                plt.show()

            train_loss = train_loss + train_loss_batch.item()
            train_loss_batch.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

        t2 = time.time()
        train_time = t2-t1

        # if test_loader:
        #     if not (step%10): # run validation every 10 steps
        #         for x, t_eval in iter(test_loader):
        #             # run test data
        #             t_eval = t_eval[0,:horizon]

        #             test_x_hat = odeint(model, x[:, :, 0], t_eval, method='rk4')
        #             test_loss_batch = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:],w)
        #             test_loss = test_loss + test_loss_batch.item()
        #         test_time = time.time()-t2
        #         print('epoch {:4d} | train time {:.2f} | train loss {:12e} | test loss {:8e} | test time {:.2f}  '
        #               .format(step, train_time, train_loss, test_loss,test_time))
        #         stats['test_loss'].append(test_loss)
        #     else:
        #         print('epoch {:4d} | train time {:.2f} | train loss {:12e} '
        #               .format(step, train_time, train_loss))
        # else:
        #     print('epoch {:4d} | train time {:.2f} | train loss {:12e} '
        #           .format(step, train_time, train_loss))

        print('epoch {:4d} | train time {:.2f} | train loss {:12e} '
              .format(step, train_time, train_loss))
        # logging
        stats['train_loss'].append(train_loss)

    return stats
