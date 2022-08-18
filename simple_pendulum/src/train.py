import torch
import time

from torchdiffeq import odeint as odeint

from .models import *
from .data import *
from .trajectories import *

def L2_loss(u, v, w = False, dim = (0,1), param = 'L2'):
    # u nominal
    # v approximate
    # u and v expected with shape : [time_steps, batch_size , (q1,p1)] 
    if param == 'L2':
        # mean only over time steps and batch_size 
        loss = (((u-v)).pow(2)).mean(dim = dim ).sum()
    elif param == 'L2weighted':
        loss = (((u-v).mul(w)).pow(2)).mean(dim = dim ).sum()
    return loss # ((u-v)*w).pow(2).mean()


def select_horizon_list(step, horizon_list = [50,100,150,200,250,300], switch_steps = [200,200,200,150,150,150]):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    # throw error if horizon_list and switch_steps not of the same length
    assert len(horizon_list)==len(switch_steps), ' horizon_list and switch_steps must have same length'

    if step <switch_steps[0]:
        horizon = horizon_list[0]
        if step==0:
            print('horizon length :', horizon)
    elif step < sum(switch_steps):
        for i in range(1,len(switch_steps)):
            if (step >= sum(switch_steps[0:i]))&(step < sum(switch_steps[0:i+1])):
                horizon = horizon_list[i]
                if step==sum(switch_steps[0:i]):
                    print('horizon length :', horizon)
    else:
        horizon = horizon_list[-1]
    return horizon 


def multilevel_strategy_update(device, step, model, resnet_config, switch_steps):
    '''

    '''
    # resnet strategy 1 :
    if resnet_config == 1 or resnet_config == 3:
        if step < sum(switch_steps[:1]):
            if step == 0:
                print('Model size increased')
            model.H_net.resblock_list = [0]
        elif step == sum(switch_steps[:1]) and len(model.H_net.resblocks) >= 2:
            print('Model size increased')
            model.H_net.resblock_list = [0, 1]
        elif step == sum(switch_steps[:2]) and len(model.H_net.resblocks) >= 3:
            print('Model size increased')
            model.H_net.resblock_list = [0, 1, 2]
        elif step == sum(switch_steps[:3]) and len(model.H_net.resblocks) >= 4:
            print('Model size increased')
            model.H_net.resblock_list = [0, 1, 2, 3]
        elif step == sum(switch_steps[:4]) and len(model.H_net.resblocks) >= 5:
            print('Model size increased')
            model.H_net.resblock_list = [0, 1, 2, 3, 4]
        elif step == sum(switch_steps[:5]) and len(model.H_net.resblocks) >= 6:
            print('Model size increased')
            model.H_net.resblock_list = [0, 1, 2, 3, 4, 5]

    # resnet strategy 2 :
    if resnet_config == 2:
        if step < sum(switch_steps[:1]):  # init : [0,1,2,3,4,5,6,7,8,9,10,11]
            if step == 0:
                print('Model size increased')
            model.H_net.resblock_list = [0, 16]
            model.H_net.alpha = torch.tensor(
                [1/len(model.H_net.resblock_list)], device=device)
        elif step == sum(switch_steps[:1]) and len(model.H_net.resblocks) >= 4:
            print('Model size increased')
            model.H_net.init_new_resblocks(0, 8, 16)
            model.H_net.resblock_list = [0, 8, 16]
            model.H_net.alpha = torch.tensor(
                [1/len(model.H_net.resblock_list)], device=device)
        elif step == sum(switch_steps[:2]) and len(model.H_net.resblocks) >= 6:
            print('Model size increased')
            model.H_net.init_new_resblocks(0, 4, 8)
            model.H_net.init_new_resblocks(8, 12, 16)
            model.H_net.resblock_list = [0, 4, 8, 12, 16]
            model.H_net.alpha = torch.tensor(
                [1/len(model.H_net.resblock_list)], device=device)
        elif step == sum(switch_steps[:3]) and len(model.H_net.resblocks) >= 8:
            print('Model size increased')
            model.H_net.init_new_resblocks(0, 2, 4)
            model.H_net.init_new_resblocks(4, 6, 8)
            model.H_net.init_new_resblocks(8, 10, 12)
            model.H_net.init_new_resblocks(12, 14, 16)
            model.H_net.resblock_list = [0, 2, 4, 6, 8, 10, 12, 14, 16]
            model.H_net.alpha = torch.tensor(
                [1/len(model.H_net.resblock_list)], device=device)
        elif step == sum(switch_steps[:4]) and len(model.H_net.resblocks) >= 10:
            print('Model size increased')
            model.H_net.init_new_resblocks(0, 1, 2)
            model.H_net.init_new_resblocks(2, 3, 4)
            model.H_net.init_new_resblocks(4, 5, 6)
            model.H_net.init_new_resblocks(8, 9, 10)
            model.H_net.init_new_resblocks(10, 11, 12)
            model.H_net.init_new_resblocks(11, 12, 13)
            model.H_net.init_new_resblocks(12, 13, 14)
            model.H_net.init_new_resblocks(14, 15, 16)
            model.H_net.resblock_list = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            model.H_net.alpha = torch.tensor(
                [1/len(model.H_net.resblock_list)], device=device)
    return model


def train(model, device, Ts, train_loader, test_loader, w, resnet_config = False, horizon=False, horizon_type=False,
          horizon_list = [50,100,150,200,250,300], 
          switch_steps = [200,200,200,150,150,150],
          epochs = 20, 
          loss_type = 'L2weighted'):
    '''
    Description:

    Inputs:

    Outpus:  
    '''
    ## TODO : make a function for one epoch
    ## first training steps on smaller amount of time_steps

    
    lr = 1e-3

    optim = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4) # Adam

    logs = {'train_loss': [], 'test_loss': []}
    # weight for loss function

    for step in range(epochs):
        
        train_loss = 0
        test_loss = 0
        t1 = time.time()

        if horizon_type == 'schedule':
            pass
        #     horizon, num_steps= select_horizon_wschedule(step,optim,
        #                                                   epoch_number,
        #                                                   switch_steps)
        #     if num_steps :
        #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_steps)
        elif horizon_type == 'auto': 
            horizon = select_horizon_list(step, horizon_list, switch_steps)
        # increase the model size and initialie the new parameters
        if resnet_config:
            model = multilevel_strategy_update(device, step, model, resnet_config, switch_steps) 
        model.train()
        for x, t_eval in iter(train_loader): 
            # x is [batch_size,(q1,p1,q2,p1),time_steps]

            t_eval = t_eval[0,:horizon]
            
            train_x_hat = odeint(model, x[:, 0, :], t_eval, method='rk4', options=dict(step_size=Ts))            
            # train_x_hat is [time_steps, batch_size, (q1,p1,q2,p1)] 

            train_loss_mini = L2_loss(torch.permute(x[:, :horizon, :], (1,0,2)) , train_x_hat[:horizon,:,:], w, param = loss_type)
            # after permute x is [time_steps, batch_size, (q1,p1,q2,p1),]

            # loss(u, v, w = False, dim = (0,1), param = loss_type)
            # if not step%10:
            #     t_plot = time.time()
            #     training_plot(t_eval, train_x_hat[:horizon,:,:], x[:, :, :horizon])
            #     print('plot time :' , time.time()-t_plot)

            train_loss = train_loss + train_loss_mini.item()

            train_loss_mini.backward() 
            optim.step() 
            optim.zero_grad()

            # if (horizon == 'schedule') and do_step:
            #   scheduler.step()


        t2 = time.time()
        train_time = t2-t1

        model.eval()
        if test_loader: 
            with torch.no_grad(): # we won't need gradients for testing
                if not (step%10): # run validation every 10 steps
                    for x, t_eval in iter(test_loader):
                        # run test data
                        t_eval = t_eval[0,:horizon]
      
                        test_x_hat = odeint(model, x[:, 0, :], t_eval, method='rk4', options=dict(step_size=Ts))      
                        #test_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:],w)
                        test_loss_mini = L2_loss(torch.permute(x[:, :horizon, :], (1,0,2)) , test_x_hat[:horizon,:,:], w, param = loss_type)
              

                        test_loss = test_loss + test_loss_mini.item()
                    test_time = time.time()-t2
                    print('epoch {:4d} | train time {:.2f} | train loss {:8e} | test loss {:8e} | test time {:.2f}  '
                          .format(step, train_time, train_loss, test_loss,test_time))
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


def load_data_device(time_steps, num_trajectories, device, batch_size, proportion, shuffle, Ts, y0, 
                                noise_std, C, m, g, l, u_func, g_func, coord_type = 'hamiltonian'):
    # create trajectories
    q, p, t_eval,_,_ = multiple_trajectories(time_steps=time_steps, num_trajectories=num_trajectories, device=device, Ts=Ts, y0=y0, 
                                noise_std=noise_std, C=C, m=m, g=g, l=l, u_func=u_func, g_func=g_func, coord_type = coord_type)

    q=q.to(device) 
    p=p.to(device) 
    t_eval=t_eval.to(device) 

    # dataloader to load data in batches
    train_loader, test_loader = data_loader(q, p, t_eval, batch_size, shuffle = shuffle, proportion = proportion)
    return train_loader, test_loader