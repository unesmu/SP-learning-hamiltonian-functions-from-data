from os import statvfs_result
import torch
import time
from .plots import *

def L2_loss(u, v, w = False, dim = (0,1), param = 'L2'):
    # u nominal
    # v approximate
    # u and v expected with shape : [time_steps, batch_size , (q1,p1,q2,p1)] 

    if param == 'L2weighted':
        # mean only over time steps and batch_size 
        loss = (((u-v)).pow(2)).mul(w).mean(dim = dim ).sum()
    elif param == 'L2':
        loss = (((u-v)).pow(2)).mean(dim = dim ).sum()
    elif param == 'L2weighted2':
        loss = (((u-v).mul(w)).pow(2)).mean(dim = dim ).sum()
    elif param == 'L2normalized':
        # formula : (u-v)/v
        # +1e-10 for stability in case of a zero # still not enough because loss on p1 explodes 
        # when nominal close to zero and predicted is far from zero
        loss = ((u.sub(v).div(u+1e-10)).pow(2)).mean(dim = dim ).sum()
    elif param == 'L2normalizedfixed':
        loss = ((u[:,:,:3].sub(v[:,:,:3]).div(u[:,:,:3]+1e-10)).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[:,:,3].sub(v[:,:,3])).pow(2)).mean(dim = dim ).sum()
    elif param == 'L2normalizedfixed2':
        # loss = ((u[1:,:,0].sub(v[1:,:,0]).div(u[1:,:,0]+1e-10)).mul(w[0]).pow(2)).mean(dim = dim ).sum()
        # loss = loss + ((u[1:,:,2].sub(v[1:,:,2]).div(u[1:,:,2]+1e-10)).mul(w[2]).pow(2)).mean(dim = dim ).sum()
        loss = ((u[1:,:,0].sub(v[1:,:,0]).div(w[0])).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,2].sub(v[1:,:,2]).div(w[2])).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,1].sub(v[1:,:,1]).div(w[1])).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,3].sub(v[1:,:,3]).div(w[3])).pow(2)).mean(dim = dim ).sum()

    elif param == 'L2normalizedfixed3':
        loss =        ((u[1:,:,0].sub(v[1:,:,0]).div(u[1:,:,0]+1e-5)).mul(w[0]).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,1].sub(v[1:,:,1]).mul(w[1])                     ).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,2].sub(v[1:,:,2]).div(u[1:,:,2]+1e-5)).mul(w[2]).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,3].sub(v[1:,:,3]).mul(w[3])                     ).pow(2)).mean(dim = dim ).sum()
        # loss = (1/4)*loss 

    elif param == 'L2normalizedfixed4': # division by std per trajectory bad some trajectory has tiny std that makes it explode
        std = u.std(dim=0)
        loss =        ((u[1:,:,0].sub(v[1:,:,0]).div(u[1:,:,0]+1e-10)).mul(w[0]).pow(2)).mean(dim = dim ).sum() 
        loss = loss + ((u[1:,:,1].sub(v[1:,:,1]).div(std[:,1]).mul(w[1])).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,2].sub(v[1:,:,2]).div(std[:,2]).mul(w[2])).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,3].sub(v[1:,:,3]).mul(w[3])).pow(2)).mean(dim = dim ).sum()

    elif param == 'L2normalizedfixed5':
        std = u.std(dim=0)
        loss =        ((u[1:,:,0].sub(v[1:,:,0]).div(std[:,0]).mul(w[0])).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,1].sub(v[1:,:,1]).div(std[:,1]).mul(w[1])).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,2].sub(v[1:,:,2]).div(std[:,2]).mul(w[2])).pow(2)).mean(dim = dim ).sum()
        loss = loss + ((u[1:,:,3].sub(v[1:,:,3]).mul(w[3])).pow(2)).mean(dim = dim ).sum()

    elif param == 'minmax': 
        std = u.std(dim=0)
        loss = 0
        for i in range(3):
            loss_t =  (u[1:,:,i].sub(v[1:,:,i])).mul(w[i])
            min = loss_t.min()
            max = loss_t.max()
            loss_t = (loss_t.div(max-min).pow(2)).mean(dim = dim ).sum()
            loss = loss + loss_t

        loss4 =  ((u[1:,:,3].sub(v[1:,:,3]).mul(w[3])).pow(2)).mean(dim = dim ).sum()  

    return loss


def select_horizon_list(step, epoch_number, horizon_list = [50,100,150,200,250,300], switch_steps = [200,200,200,150,150,150]):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    # throw error if 
    assert len(horizon_list)==len(switch_steps), ' horizon_list and switch_steps must have same length'

    if step <=switch_steps[0]:
        horizon = horizon_list[0]
        if step==0:
            print('horizon length :', horizon)

    for i in range(len(switch_steps)):
      if (step > sum(switch_steps[0:i]))&(step <= sum(switch_steps[0:i+1])):
          horizon = horizon_list[i]
          if step==sum(switch_steps[0:i])+1:
              print('horizon length :', horizon)

    if step > sum(switch_steps):
        horizon = horizon_list[-1]
        if step== sum(switch_steps)+1:
            print('horizon length :', horizon)
    return horizon 

def select_horizon_wschedule(step,optim,epoch_number, 
                             switch_steps = (200,200,150,150,150)       
                            ):
    '''
    Description:

    Inputs:

    Outpus:
      
    '''
    n1, n2, n3, n4, n5 = switch_steps
    # n1 number of steps with a horizon =  50
    # n2 number of steps with a horizon =  100
    # n3 number of steps with a horizon =  150
    # n4 number of steps with a horizon =  200
    # n5 number of steps with a horizon =  250
    # horizon = 300 for the remainder of training steps 

    if step <=n1:
        horizon = 50
        if step==0:
            print('horizon length :', horizon)
            # steps = n1
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
        num_steps = 0   
        do_step = False
            
    elif (step > n1)&(step <= n1+n2):
        horizon = 100
        if step==n1+1:
            print('horizon length :', horizon)
            # steps = n2
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
        num_steps = 0   
        do_step = False
    elif (step > n1+n2)&(step <= n1+n2+n3):
        horizon = 150
        if step==n1+n2+1:
            print('horizon length :', horizon)
            # steps = n3
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
        num_steps = 0   
        do_step = False
    elif (step > n1+n2+n3)&(step <= n1+n2+n3+n4):
        horizon = 200
        if step==n1+n2+n3+1:
            print('horizon length :', horizon)
            # steps = n4
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
        num_steps = 0     
        do_step = False
    elif (step > n1+n2+n3+n4)&(step <= n1+n2+n3+n4+n5):
        horizon = 250
        if step==n1+n2+n3+n4+1:
            print('horizon length :', horizon)
            # steps = n5
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
        num_steps = 0     
        do_step = False
    else :
        horizon = 300
        do_step = True
        num_steps = 0
        if step==n1+n2+n3+n4+n5+1:
            print('horizon length :', horizon)
            num_steps = epoch_number - (n1+n2+n3+n4+n5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_steps)

    return horizon, num_steps



def train(model, Ts, train_loader, test_loader, w, epoch_number, horizon=False, horizon_type=False,
          horizon_list = [50,100,150,200,250,300], 
          switch_steps = [200,200,200,150,150,150],
          epochs = 20, 
          loss_type = 'L2'):
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
            horizon = select_horizon_list(step, epoch_number, horizon_list, switch_steps)

        model.train()
        for u, x, t_eval in iter(train_loader): # we don't really need u since the model knows u_func
            # x is [batch_size,(q1,p1,q2,p1),time_steps]
            t_eval = t_eval[0,:horizon]

            train_x_hat = odeint(model, x[:, :, 0], t_eval, method='rk4', options=dict(step_size=Ts))            
            # train_x_hat is [time_steps, batch_size, (q1,p1,q2,p1)] 
            #train_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , train_x_hat[:horizon,:,:],w)#[:,:horizon])
            train_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , train_x_hat[:horizon,:,:], w, param = loss_type)
            # after permute x is [time_steps, batch_size, (q1,p1,q2,p1),]

            # loss(u, v, w = False, dim = (0,1), param = loss_type)
            if not step%10:
                t_plot = time.time()
                training_plot(t_eval, train_x_hat[:horizon,:,:], x[:, :, :horizon])
                print('plot time :' , time.time()-t_plot)

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
                    for u, x, t_eval in iter(test_loader):
                        # run test data
                        t_eval = t_eval[0,:horizon]
      
                        test_x_hat = odeint(model, x[:, :, 0], t_eval, method='rk4', options=dict(step_size=Ts))      
                        #test_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:],w)
                        test_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:], w, param = loss_type)
              

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


def train_alternating(model, Ts, train_loader, test_loader, w, epoch_number, horizon=False, horizon_type=False,
          horizon_list = [50,100,150,200,250,300], 
          switch_steps = [200,200,200,150,150,150],
          epochs = 20, 
          loss_type = 'L2'):
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
            # horizon, num_steps= select_horizon_wschedule(step,optim,
            #                                               epoch_number,
            #                                               switch_steps)
            # if num_steps :
            #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_steps)
            pass
        elif horizon_type == 'auto': 
            horizon = select_horizon_list(step, epoch_number, horizon_list, switch_steps)

        model.train()
        for u, x, t_eval in iter(train_loader): # we don't really need u since the model knows u_func
            # x is [batch_size,(q1,p1,q2,p1),time_steps]
            t_eval = t_eval[0,:horizon]

            for i in range(2):
                if i==0:
                    model.freeze_H_net(freeze=True)
                    model.freeze_G_net(freeze=False)
                elif i==1:
                    model.freeze_H_net(freeze=False)
                    model.freeze_G_net(freeze=True)

                train_x_hat = odeint(model, x[:, :, 0], t_eval, method='rk4', options=dict(step_size=Ts))            
                # train_x_hat is [time_steps, batch_size, (q1,p1,q2,p1)] 
                #train_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , train_x_hat[:horizon,:,:],w)#[:,:horizon])
                train_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , train_x_hat[:horizon,:,:], w, param = loss_type)
                # after permute x is [time_steps, batch_size, (q1,p1,q2,p1),]

                # loss(u, v, w = False, dim = (0,1), param = loss_type)
                if (not step%10)&(i==0):

                    t_plot = time.time()
                    training_plot(t_eval, train_x_hat[:horizon,:,:], x[:, :, :horizon])
                    print('plot time :' , time.time()-t_plot)

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
                    for u, x, t_eval in iter(test_loader):
                        # run test data
                        t_eval = t_eval[0,:horizon]
      
                        test_x_hat = odeint(model, x[:, :, 0], t_eval, method='rk4', options=dict(step_size=Ts))      
                        #test_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:],w)
                        test_loss_mini = L2_loss(torch.permute(x[:, :, :horizon], (2,0,1)) , test_x_hat[:horizon,:,:], w, param = loss_type)
              

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