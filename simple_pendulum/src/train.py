import torch
import time

from torchdiffeq import odeint as odeint

from .models import *
from .data import *
from .trajectories import *
from .train_helpers import *


class Training:
    ''' 
    Training class
    '''

    def __init__(self, model, device, Ts, train_loader, test_loader, w,
                 resnet_config=False, horizon=False, horizon_type=False,
                 horizon_list=[50, 100, 150, 200, 250, 300],
                 switch_steps=[200, 200, 200, 150, 150, 150],
                 epochs=20,
                 loss_type='L2weighted'):

        pass

    def _init_model(self, model_name):

        pass

    def _output_training_stats(self, epoch, train_loss, test_loss, t0):
        """
        Output and save training stats every epoch or multiple of epochs
        """

        if epoch % self.print_every == 0 or epoch == self.num_epochs-1:
            print('[%d/%d]\t train loss: %.4f, test loss: %.4f, t: %2.3f'
                  % (epoch, self.num_epochs, train_loss, test_loss, time.time()-t0))

        return

    def _train_step(self):
        """
        basic training step of the model
        """
        return

    def _test_step(self):
        """
        basic training step of the model
        """
        return

    def train(self):
        """
        Training function. Uses the classes variables, and returns stats from the
        training procedure
        """
        return


def train(model, device, Ts, train_loader, test_loader, w, resnet_config=False, horizon=False, horizon_type=False,
          horizon_list=[50, 100, 150, 200, 250, 300],
          switch_steps=[200, 200, 200, 150, 150, 150],
          epochs=20,
          loss_type='L2weighted'):
    '''
    Description:

    Inputs:

    Outpus:  
    '''

    lr = 1e-3

    optim = torch.optim.AdamW(model.parameters(), lr,
                              weight_decay=1e-4)  # Adam

    logs = {'train_loss': [], 'test_loss': []}
    # weight for loss function

    for step in range(epochs):

        train_loss = 0
        test_loss = 0
        t1 = time.time()

        if horizon_type == 'auto':
            horizon = select_horizon_list(step, horizon_list, switch_steps)

        # increase the model size and initialie the new parameters
        if resnet_config:
            model = multilevel_strategy_update(
                device, step, model, resnet_config, switch_steps)
        
        model.train()
        for x, t_eval in iter(train_loader):
            # x is [batch_size,(q1,p1,q2,p1),time_steps]

            t_eval = t_eval[0, :horizon]

            train_x_hat = odeint(
                model, x[:, 0, :], t_eval, method='rk4', options=dict(step_size=Ts))
            # train_x_hat is [time_steps, batch_size, (q1,p1,q2,p1)]

            train_loss_mini = L2_loss(torch.permute(
                x[:, :horizon, :], (1, 0, 2)), train_x_hat[:horizon, :, :], w, param=loss_type)
            # after permute x is [time_steps, batch_size, (q1,p1,q2,p1),]


            train_loss = train_loss + train_loss_mini.item()

            train_loss_mini.backward()
            optim.step()
            optim.zero_grad()

        t2 = time.time()
        train_time = t2-t1

        model.eval()
        if test_loader:
            with torch.no_grad():  # we won't need gradients for testing
                if not (step % 10):  # run validation every 10 steps
                    for x, t_eval in iter(test_loader):
                        # run test data
                        t_eval = t_eval[0, :horizon]

                        test_x_hat = odeint(
                            model, x[:, 0, :], t_eval, method='rk4', options=dict(step_size=Ts))
                        #test_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:],w)
                        test_loss_mini = L2_loss(torch.permute(
                            x[:, :horizon, :], (1, 0, 2)), test_x_hat[:horizon, :, :], w, param=loss_type)

                        test_loss = test_loss + test_loss_mini.item()
                    test_time = time.time()-t2
                    print('epoch {:4d} | train time {:.2f} | train loss {:8e} | test loss {:8e} | test time {:.2f}  '
                          .format(step, train_time, train_loss, test_loss, test_time))
                    logs['test_loss'].append(test_loss)

                else:
                    print('epoch {:4d} | train time {:.2f} | train loss {:8e} '
                          .format(step, train_time, train_loss))
        else:
            print('epoch {:4d} | train time {:.2f} | train loss {:8e} '
                  .format(step, train_time, train_loss))

        # logging
        logs['train_loss'].append(train_loss)
    return logs



