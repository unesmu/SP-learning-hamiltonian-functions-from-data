import torch


def choose_nonlinearity(name):
    """
    From the SymODEN repository
    returns an activation function that can be evaluated
    """
    nl = None
    if name == "tanh":
        nl = torch.tanh
    elif name == "x+sin(x)^2":
        nl = lambda x: x + torch.sin(x).pow(2)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl


""" MLP """


class hidden_Layer(torch.nn.Module):
    """
    Fully connected layer with activation function
    """

    def __init__(self, hidden_dim, activation="tanh"):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation = choose_nonlinearity(activation)  # activation function

    def forward(self, x):
        x = self.activation(self.fc(x))
        return x


class MLP(torch.nn.Module):
    """
    MLP with number of hidden layers as a parameter
    """

    def __init__(
        self,
        input_dim=2,
        hidden_dim=90,
        nb_hidden_layers=4,
        output_dim=1,
        activation="x+sin(x)^2",
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.Sequential(
            *(hidden_Layer(hidden_dim, activation) for _ in range(nb_hidden_layers))
        )  # trick from EE-559 to define hidden layers
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = choose_nonlinearity(activation)  # activation function

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.hidden_layers(h)
        return self.fc2(h)


class Expanding_HNN(torch.nn.Module):
    """
    Model compose of [ MLP - RESBLOCK1 - RESBLOCK2] used in the following way:
    First we train the model with only the MLP, then we introduce RESBLOCK1 when we increase the horizon,
    but initialise RESBLOCK1 with very small weights,
    """

    def __init__(
        self,
        resblock_list,
        num_blocks=4,
        input_dim=4,
        hidden_dim=90,
        nb_hidden_layers=1,
        output_dim=1,
        activation_res="x+sin(x)^2",
        activation_mlp="x+sin(x)^2",
    ):
        super(Expanding_HNN, self).__init__()

        self.resblocks = [
            MLP(
                input_dim=output_dim,
                hidden_dim=hidden_dim,
                nb_hidden_layers=nb_hidden_layers,
                output_dim=output_dim,
                activation=activation_res,
            )
            for _ in range(num_blocks)
        ]

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nb_hidden_layers=nb_hidden_layers,
            output_dim=output_dim,
            activation=activation_mlp,
        )

        self.resblock_list = resblock_list
        self.make_params_small()  # make the resblock parameters small

    def forward(self, x):

        y = self.mlp(x)

        for i in self.resblock_list:
            y = self.resblocks[i](y) + y

        return y

    def make_params_small(self):
        with torch.no_grad():
            for i in range(len(self.resblocks)):
                for param in self.resblocks[i].parameters():
                    param.copy_(param / 1000)


""" RESNET """


class Interp_HNN(torch.nn.Module):
    """
    MLP with number of hidden layers as a parameter
    """

    def __init__(
        self,
        resblock_list,
        num_blocks=4,
        input_dim=4,
        hidden_dim=25,
        nb_hidden_layers=2,
        output_dim=1,
        activation_res="x+sin(x)^2",
        activation_mlp="x+sin(x)^2",
    ):
        super(Interp_HNN, self).__init__()

        self.resblocks = [
            MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                nb_hidden_layers=nb_hidden_layers,
                output_dim=input_dim,
                activation=activation_res,
            )
            for _ in range(num_blocks)
        ]

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nb_hidden_layers=nb_hidden_layers,
            output_dim=output_dim,
            activation=activation_mlp,
        )

        self.alpha = torch.tensor([1])

        self.resblock_list = resblock_list

    def forward(self, x):
        y = x

        for i in self.resblock_list:
            y = self.resblocks[i](y) * self.alpha + y

        y = self.mlp(y)
        return y

    def init_new_resblocks_two(self, i, j):
        # i and j are the new resblocks to be initialised
        # in our simple case we introduce reblocks 1 and 2 between resblocks 0 and 3

        with torch.no_grad():
            for param1, param2, param3, param4 in zip(
                self.resblocks[i - 1].parameters(),
                self.resblocks[i].parameters(),
                self.resblocks[j].parameters(),
                self.resblocks[j + 1].parameters(),
            ):

                param2.copy_((param1 + param2) / 2)
                param3.copy_((param3 + param4) / 2)

    def init_new_resblocks(self, i, j, k):
        # i is the new resblocks to be initialised
        # in our simple case we introduce reblocks 1 and 2 between resblocks 0 and 3

        with torch.no_grad():
            for param1, param2, param3 in zip(
                self.resblocks[i].parameters(),
                self.resblocks[j].parameters(),
                self.resblocks[k].parameters(),
            ):
                param2.copy_((param1 + param3) / 2)


class Expanding_wide_HNN(torch.nn.Module):
    """
    Model compose of [RESBLOCK1 - RESBLOCK2 - MLP] used in the following way:
    First we train the model with only the MLP, then we introduce RESBLOCK1 when we increase the horizon,
    but initialise RESBLOCK2 with very small weights,
    """

    def __init__(
        self,
        resblock_list,
        num_blocks=4,
        input_dim=4,
        hidden_dim=90,
        nb_hidden_layers=1,
        output_dim=1,
        activation_res="x+sin(x)^2",
        activation_mlp="x+sin(x)^2",
    ):
        super(Expanding_wide_HNN, self).__init__()

        self.resblocks = [
            MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                nb_hidden_layers=nb_hidden_layers,
                output_dim=input_dim,
                activation=activation_res,
            )
            for _ in range(num_blocks)
        ]

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nb_hidden_layers=nb_hidden_layers,
            output_dim=output_dim,
            activation=activation_mlp,
        )

        self.resblock_list = resblock_list
        self.make_params_small()  # make the resblock parameters small

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
                    param.copy_(param / 1000)
