from addl_utils import *
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Represents the actor network.
    Returns the action given the state.
    """


    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        hidden_dim = 256
        # input layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # hidden layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Save the max action
        self.max_action = max_action

        # Initialize the weights and biases
        self.reset_parameters()

    def forward(self, state):

        # Forward pass
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        # Scale the output to the action space
        x = self.max_action * x
        return x

    def reset_parameters(self):

        # Initialize the weights and biases
        variance_initializer_(self.fc1.weight, scale=1. / 3., 
                                mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.fc1.bias)
        variance_initializer_(self.fc2.weight, scale=1. / 3., 
                                mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        torch.nn.init.zeros_(self.fc3.bias)



class Critic(nn.Module):
    """
    Represents the critic network.
    Returns the Q-value given the state and action.
    """

    def __init__(self, state_dim, action_dim, output_dim=1):
        super().__init__()

        hidden_dim = 256
        # input layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # hidden layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Initialize the weights and biases
        self.reset_parameters()

    def forward(self, state, action):

        # Forward pass
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)

        return x
    

    def reset_parameters(self):

        # Initialize the weights and biases
        variance_initializer_(self.fc1.weight, scale=1. / 3., 
                                mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.fc1.bias)
        variance_initializer_(self.fc2.weight, scale=1. / 3., 
                                mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        torch.nn.init.zeros_(self.fc3.bias)