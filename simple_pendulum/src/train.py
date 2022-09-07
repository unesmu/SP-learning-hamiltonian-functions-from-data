from asyncio import wait_for
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
                 PATH,
                 device,
                 w=[1.0, 1.0],
                 resnet_config=False,
                 horizon=False,
                 horizon_type=False,
                 horizon_list=[50, 100, 150, 200, 250, 300],
                 switch_steps=[200, 200, 200, 150, 150, 150],
                 epoch_num=None,
                 loss_type='L2weighted',
                 test_every=10,
                 print_every=10,
                 lr=1e-3,
                 weight_decay=1e-4,
                 model_name='Input_HNN_chirp',
                 time_steps=200,
                 num_trajectories=125,
                 batch_size=100,
                 proportion=0.8,
                 utype=None,
                 gtype=None,
                 shuffle=False,
                 coord_type='hamiltonian',
                 save_suffix='',
                 ):

        self.device = device
        self.w = w
        self.resnet_config = resnet_config
        self.horizon = horizon
        self.horizon_type = horizon_type
        self.horizon_list = horizon_list
        self.switch_steps = switch_steps
        self.loss_type = loss_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_every = test_every
        self.print_every = print_every

        self.model_name = model_name
        self.y0, self.noise_std, self.Ts, self.C, self.m, self.g, self.l = simple_pendulum_parameters()

        self.time_steps = time_steps
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.proportion = proportion
        self.utype = utype
        self.gtype = gtype
        self.shuffle = shuffle
        self.coord_type = coord_type

        self.u_func = U_FUNC(utype=utype)
        self.g_func = G_FUNC(device, gtype=gtype)

        self.w = torch.tensor(w, device=self.device)
        if epoch_num is None:
            self.epoch_num = sum(switch_steps)
        print('Number of training epochs: ', self.epoch_num)

        print('Generating dataset')
        self._init_data_loaders()
        print('Dataset created')
        self.model_path, self.plot_path = create_paths(PATH,
                                                       save_suffix,
                                                       model_name)
        print('Paths created')
        self._init_model()
        print('Model initiatialized. \n Number of parameters :', count_parameters(self.model))

    def _init_data_loaders(self):
        self.train_loader, self.test_loader = load_data_device(self.time_steps, self.num_trajectories,
                                                               self.device, self.batch_size, self.proportion,
                                                               self.shuffle, self.Ts, self.y0, self.noise_std,
                                                               self.C, self.m, self.g, self.l, self.u_func,
                                                               self.g_func, self.coord_type)

    def _init_model(self):
        if self.model_name == 'Input_HNN_chirp':

            H_net = MLP(input_dim=2, hidden_dim=60, nb_hidden_layers=2,
                        output_dim=1, activation='x+sin(x)^2')
            self.model = Input_HNN(
                u_func=self.u_func, G_net=self.g_func, H_net=H_net, device=self.device)

        elif self.model_name == 'Simple_HNN_':

            H_net = MLP(input_dim=2, hidden_dim=60, nb_hidden_layers=2,
                        output_dim=1, activation='x+sin(x)^2')
            self.model = Simple_HNN(
                H_net=H_net, device=self.device, dissip=False)


        self.model.to(self.device)


    def _output_training_stats(self, step, train_loss, test_loss, train_time, test_time):
        """
        Output and save training stats every epoch or multiple of epochs
        """

        if step % self.print_every == 0 or step == self.epoch_num-1:
            if step % self.test_every == 0:
                print('[%3d/%3d]\t train loss: %4e, t_train: %2.2f, test loss: %4e, t_test: %2.2f'
                      % (step, self.epoch_num, train_loss, train_time, test_loss, test_time))
                self.test_epochs.append(step)
            else:
                print('[%3d/%3d]\t train loss: %4e, t_train: %2.2f'
                      % (step, self.epoch_num, train_loss, train_time))

    def _train_step(self, train_loss, x, t_eval):
        """
        basic training step of the model
        """

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

    def _test_step(self, test_loss, x, t_eval):
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
                                       weight_decay=self.weight_decay)  # Adam

        logs = {'train_loss': [], 'test_loss': []}

        self.test_epochs = []

        for step in range(self.epoch_num):

            test_loss = 0
            train_loss = 0

            t1 = time.time()

            if self.horizon_type == 'auto':
                self.horizon = select_horizon_list(
                    step, self.horizon_list, self.switch_steps)

            # increase the model size and initialise the new parameters
            if self.resnet_config:
                self.model = multilevel_strategy_update(
                    self.device, step, self.model, self.resnet_config, self.switch_steps)

            self.model.train()

            for x, t_eval in iter(self.train_loader):
                train_loss = self._train_step(train_loss, x, t_eval)
            
            logs['train_loss'].append(train_loss)

            t2 = time.time()
            train_time = t2-t1

            self.model.eval()

            if self.test_loader:
                with torch.no_grad():  # we won't need gradients for testing
                    if step % self.test_every == 0:  # run validation every 10 steps
                        for x, t_eval in iter(self.test_loader):
                            test_loss = self._test_step(test_loss, x, t_eval)
                        
                        logs['test_loss'].append(test_loss)

            test_time = time.time()-t2

            self._output_training_stats(
                step, train_loss, test_loss, train_time, test_time)

            
            
            

        logs['test_epochs'] = self.test_epochs

        self.logs = logs
        return logs
