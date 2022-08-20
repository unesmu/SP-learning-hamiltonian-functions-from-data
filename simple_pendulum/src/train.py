import torch
import time

from torchdiffeq import odeint as odeint

from .dynamics import *
from .data import *
from .models_main import *
from .models_sub import *
from .plots import *
from .train import *
from .train_helpers import *
from .utils import *

class Training:
    ''' 
    Training class
    '''

    def __init__(self, 
                 device, 
                 Ts, 
                 train_loader, 
                 test_loader, 
                 w,
                 resnet_config=False, 
                 horizon=False, 
                 horizon_type=False,
                 horizon_list=[50, 100, 150, 200, 250, 300],
                 switch_steps=[200, 200, 200, 150, 150, 150],
                 num_epochs=20,
                 loss_type='L2weighted', 
                 test_every = 10,
                 lr = 1e-3,
                 weight_decay = 1e-4,
                 model_name = 'Input_HNN_chirp'):

        
        self.device = device
        self.Ts = Ts
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.w = w
        self.resnet_config = resnet_config
        self.horizon = horizon
        self.horizon_type = horizon_type
        self.horizon_list = horizon_list
        self.switch_steps = switch_steps
        self.num_epochs = num_epochs
        self.loss_type = loss_type
        self.lr = lr
        self.weight_decay= weight_decay
        self.test_every = test_every
        self.model_name = model_name

    def _init_model(self):
        if self.model_name == 'Input_HNN_chirp':

            H_net = MLP(input_dim=2, hidden_dim=60, nb_hidden_layers=2, output_dim=1, activation='x+sin(x)^2')
            self.model = Input_HNN(u_func=self.u_func, G_net=self.g_func, H_net=H_net, device=self.device)
            self.model.to(self.device)

        # self.model = model
        pass

    def _output_training_stats(self, step, train_loss, test_loss, train_time, test_time):
        """
        Output and save training stats every epoch or multiple of epochs
        """

        if step % self.print_every == 0 or step == self.num_epochs-1:
            if step % self.test_every == 0:
                print('[%3d/%3d]\t train loss: %8e, test loss: %8e, t_train: %2.3f, t_test: %2.3f'
                    % (step, self.num_epochs, train_loss, test_loss, train_time,test_time))
            else : 
                print('[%3d/%3d]\t train loss: %8e, t_train: %2.3f'
                    % (step, self.num_epochs, train_loss, train_time))
 
        return

    def _train_step(self, x, t_eval):
        """
        basic training step of the model
        """
        train_loss = 0

        # x is [batch_size,(q1,p1,q2,p1),time_steps]

        t_eval = t_eval[0, :self.horizon]

        train_x_hat = odeint(
            self.model, x[:, 0, :], t_eval, method='rk4', options=dict(step_size=self.Ts))
        # train_x_hat is [time_steps, batch_size, (q1,p1,q2,p1)]

        train_loss_mini = L2_loss(torch.permute(
            x[:, :self.horizon, :], (1, 0, 2)), train_x_hat[:self.horizon, :, :], self.w, param=self.loss_type)
        # after permute x is [time_steps, batch_size, (q1,p1,q2,p1),]


        train_loss = train_loss + train_loss_mini.item()

        train_loss_mini.backward()
        self.optim.step()
        self.optim.zero_grad()

        return train_loss

    def _test_step(self, step, x, t_eval):
        """
        basic training step of the model
        """

        # run test data
        t_eval = t_eval[0, :self.horizon]

        test_x_hat = odeint(
            self.model, x[:, 0, :], t_eval, method='rk4', options=dict(step_size=self.Ts))
        #test_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:],w)
        test_loss_mini = L2_loss(torch.permute(
            x[:, :self.horizon, :], (1, 0, 2)), test_x_hat[:self.horizon, :, :], self.w, param=self.loss_type)

        test_loss = test_loss + test_loss_mini.item()

        return test_loss

    def train(self):
        """
        Training function. Uses the classes variables, and returns stats from the
        training procedure
        """


        self.optim = torch.optim.AdamW(self.model.parameters(), self.lr,
                                weight_decay= self.weight_decay)  # Adam

        logs = {'train_loss': [], 'test_loss': []}


        for step in range(self.epochs):

            test_loss = 0

            t1 = time.time()

            if self.horizon_type == 'auto':
                self.horizon = select_horizon_list(step, self.horizon_list, self.switch_steps)

            # increase the model size and initialise the new parameters
            if self.resnet_config:
                self.model = multilevel_strategy_update(
                    self.device, step, self.model, self.resnet_config, self.switch_steps)
            
            self.model.train()

            for x, t_eval in iter(self.train_loader):
                train_loss = self._train_step(x, t_eval)

            t2 = time.time()
            train_time = t2-t1

            self.model.eval()

            if self.test_loader:

                with torch.no_grad():  # we won't need gradients for testing
                    if step % self.test_every == 0:  # run validation every 10 steps
                        for x, t_eval in iter(self.test_loader):
                            test_loss = self._test_step(step, x, t_eval)
            
            test_time = time.time()-t2

            self._output_training_stats( step, train_loss, test_loss, train_time, test_time)

            # logging
            logs['train_loss'].append(train_loss)
            logs['test_loss'].append(test_loss)

        return logs
















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



