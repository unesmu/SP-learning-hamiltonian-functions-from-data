import torch

""" MLP MODELS """

def choose_nonlinearity(name):
    '''
    From the SymODEN repository

    Description:

    Inputs:

    Outpus:
      
    '''
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'x+sin(x)^2':
        nl = lambda x: x + torch.sin(x).pow(2)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl

class hidden_Layer(torch.nn.Module):
    '''
    Fully connected layer with activation function
    '''
    def __init__(self,hidden_dim, activation='tanh'):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_dim,hidden_dim)
        self.activation = choose_nonlinearity(activation)# activation function

    def forward(self, x):
        x=self.activation(self.fc(x))
        return x


class MLP(torch.nn.Module):
    '''
    MLP with number of hidden layers as a parameter
    '''
    def __init__(self, input_dim=2, hidden_dim=90, nb_hidden_layers=4, output_dim=1, activation='x+sin(x)^2'):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.Sequential(
            *(hidden_Layer(hidden_dim, activation)
              for _ in range(nb_hidden_layers))
        ) # trick from EE-559 to define hidden layers
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = choose_nonlinearity(activation) # activation function

    def forward(self, x):
        h = self.activation( self.fc1(x) )
        h = self.hidden_layers(h) 
        return self.fc2(h)

class Expanding_HNN(torch.nn.Module):
    '''
    Model compose of [ MLP - RESBLOCK1 - RESBLOCK2] used in the following way:
    First we train the model with only the MLP, then we introduce RESBLOCK1 when we increase the horizon, 
    but initialise RESBLOCK1 with very small weights, 
    '''
    def __init__(self, resblock_list, num_blocks =4, input_dim=4, hidden_dim=90, nb_hidden_layers=1, output_dim=1, activation_res='x+sin(x)^2', activation_mlp='x+sin(x)^2'):
        super(Expanding_HNN, self).__init__()
        
        self.resblocks = [MLP(input_dim=output_dim, hidden_dim=hidden_dim, nb_hidden_layers=nb_hidden_layers,
                                 output_dim=output_dim, activation=activation_res) for _ in range(num_blocks)]

        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, nb_hidden_layers=nb_hidden_layers,
                                 output_dim=output_dim, activation=activation_mlp)
        
        self.resblock_list = resblock_list
        self.make_params_small() # make the resblock parameters small
        
    def forward(self, x):

        y = self.mlp(x)

        for i in self.resblock_list:
            y = self.resblocks[i](y) + y

        return y
    
    def make_params_small(self):
        with torch.no_grad():
            for i in range(len(self.resblocks)):
                for param in self.resblocks[i].parameters():
                    param.copy_(param/1000)


""" RESNET MODELS """

class Interp_HNN(torch.nn.Module):
    '''
    MLP with number of hidden layers as a parameter
    '''

    def __init__(self, resblock_list, num_blocks=4, input_dim=4, hidden_dim=25, nb_hidden_layers=2, output_dim=1, activation_res='x+sin(x)^2', activation_mlp='x+sin(x)^2'):
        super(Interp_HNN, self).__init__()

        self.resblocks = [MLP(input_dim=input_dim, hidden_dim=hidden_dim, nb_hidden_layers=nb_hidden_layers,
                              output_dim=input_dim, activation=activation_res) for _ in range(num_blocks)]

        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, nb_hidden_layers=nb_hidden_layers,
                       output_dim=output_dim, activation=activation_mlp)

        self.alpha = torch.tensor([1])

        self.resblock_list = resblock_list

    def forward(self, x):
        y = x

        for i in self.resblock_list:
            y = self.resblocks[i](y)*self.alpha + y

        y = self.mlp(y)
        return y

    def init_new_resblocks_two(self, i, j):
        # i and j are the new resblocks to be initialised
        # in our simple case we introduce reblocks 1 and 2 between resblocks 0 and 3

        with torch.no_grad():
            for param1, param2, param3, param4 in zip(self.resblocks[i-1].parameters(), self.resblocks[i].parameters(), self.resblocks[j].parameters(), self.resblocks[j+1].parameters()):

                param2.copy_((param1 + param2)/2)
                param3.copy_((param3 + param4)/2)

    def init_new_resblocks(self, i, j, k):
        # i is the new resblocks to be initialised
        # in our simple case we introduce reblocks 1 and 2 between resblocks 0 and 3

        with torch.no_grad():
            for param1, param2, param3 in zip(self.resblocks[i].parameters(), self.resblocks[j].parameters(), self.resblocks[k].parameters()):
                param2.copy_((param1 + param3)/2)

class Expanding_wide_HNN(torch.nn.Module):
    '''
    Model compose of [RESBLOCK1 - RESBLOCK2 - MLP] used in the following way:
    First we train the model with only the MLP, then we introduce RESBLOCK1 when we increase the horizon, 
    but initialise RESBLOCK2 with very small weights, 
    '''
    def __init__(self, resblock_list, num_blocks =4, input_dim=4, hidden_dim=90, nb_hidden_layers=1, output_dim=1, activation_res='x+sin(x)^2', activation_mlp='x+sin(x)^2'):
        super(Expanding_wide_HNN, self).__init__()
        
        self.resblocks = [MLP(input_dim=input_dim, hidden_dim=hidden_dim, nb_hidden_layers=nb_hidden_layers,
                                 output_dim=input_dim, activation=activation_res) for _ in range(num_blocks)]

        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, nb_hidden_layers=nb_hidden_layers,
                                 output_dim=output_dim, activation=activation_mlp)
        
        self.resblock_list = resblock_list
        self.make_params_small() # make the resblock parameters small
        
    def forward(self, x):

        y = x

        for i in self.resblock_list:
            y = self.resblocks[i](y) + y
        y = self.mlp(y)
        return y
    
    def make_params_small(self):
        with torch.no_grad():
            for i in range(len(self.resblocks)):
                for param in self.resblocks[i].parameters():
                    param.copy_(param/1000)


""" SIMPLE  HNN """

class Simple_HNN(torch.nn.Module):
    '''
    Modified version of the original SymODEN_R module
    Equivalent of unconstrained ODE HNN from the report
    Architecture for input (q, p, u), 
    where q and p are tensors of size (bs, n) and u is a tensor of size (bs, 1)
    '''
    def __init__(self, H_net=None,  device=None, dissip=False):
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

            q, p = torch.chunk(x,2,dim=-1)

            q_p.requires_grad_(True)

            H = self.H_net(q_p)

            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)

            dH = dH[0]

            dHdq , dHdp = torch.chunk(dH,2,dim=-1)

            dqdt = dHdp 
            if self.dissip:
                dpdt = -dHdq - self.C_dissip*dHdp
            else : 
                dpdt = -dHdq
            
            # symplectic gradient 
            S_h = torch.cat((dqdt, dpdt), 
                            dim=-1) 
            return S_h 



""" INPUT HNN """


class Input_HNN(torch.nn.Module):
    '''
    Modified version of the original SymODEN_R module from symoden repository
    Similar to unconstrained ODE HNN from the report

    '''
    def __init__(self, u_func = None, G_net=None, H_net=None, device=None, dissip=False):
        super(Input_HNN, self).__init__()
        self.H_net = H_net
        self.G_net = G_net
        self.u_func = u_func

        self.device = device
        self.dissip = dissip
        # add learnable dissipation coefficients
        self.C = torch.nn.Parameter(torch.tensor([0.02]).sqrt()) # torch.nn.Parameter(torch.randn(1)+1) # torch.nn.Parameter(torch.tensor([0.5])) 
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

            dHdq, dHdp= torch.chunk(dH,2,dim=-1)
            

            G = self.G_net.forward(q_p)

            if self.u_func:
                u = self.u_func.forward(t)
            else:
                u = torch.tensor([0.0])
            dqdt = dHdp + (G[:,0].T*u).unsqueeze(dim=1) 
            if self.dissip:
                dpdt = -dHdq + (G[:,1].T*u).unsqueeze(dim=1)  - self.C.pow(2)*dHdp
            else:
                dpdt = -dHdq + (G[:,1].T*u).unsqueeze(dim=1)
            # symplectic gradient 
            S_h = torch.cat((dqdt, dpdt), 
                            dim=-1) 
            
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
            param.requires_grad = (not freeze)

    def freeze_H_net(self, freeze=True): 
        """
        Only freez the H_net parameters
        Inputs
            freeze(bool) : True = freeze the parameters; False = don't
      
        outputs:
            None
        """
        for param in self.H_net.parameters(): 
            param.requires_grad = (not freeze)