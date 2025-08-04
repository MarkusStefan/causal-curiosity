from abc import ABC, abstractmethod
from torch import nn
import numpy as np


class Policy(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, state, *args, **kwargs):
        """
        This method should be overridden by subclasses to implement the policy logic.
        
        :param state: The current state of the environment.
        :return: The action to be taken.
        """
        return self.act(state, *args, **kwargs)
    
    @abstractmethod
    def act(self, state, *args, **kwargs):
        """
        This method should be overridden by subclasses to define how the policy acts given a state.
        
        :param state: The current state of the environment.
        :return: The action to be taken.
        """
        pass




class RandomPolicy(Policy):
    def __init__(self, action_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = action_space

    def act(self, state, *args, **kwargs):
        """
        Select a uniformly random action from the action space.
        
        :param state: The current state of the environment (not used in this policy).
        :return: A random action.
        """
        return (
            np.random.choice(self.action_space) 
            if isinstance(self.action_space, list) 
            else self.action_space.sample()
        )




class PolicyMLP(Policy, nn.Module):
    """ A policy that uses a machine learning model to predict actions based on the current state.
    This class is a wrapper around a machine learning model, allowing it to be used as a policy in reinforcement learning environments.
    It inherits from both Policy and nn.Module, enabling it to be used in PyTorch-based environments.
    """
    def __init__(self, state_space, action_space, *args, **kwargs):
        """
        Initialize the policy with a machine learning model.
        
        :param model: The machine learning model to use for action selection.
        """
        super().__init__(*args, **kwargs)
        self.state_space = state_space
        self.action_space = action_space
        
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_space)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
        )

    def forward(self, state):
        """
        Forward pass through the model to predict the action based on the current state.
        
        :param state: The current state of the environment.
        :return: The predicted action.
        """
        return self.model(state)

    def act(self, state, *args, **kwargs):
        """
        Use the model to predict the action based on the current state.
        
        :param state: The current state of the environment.
        :return: The predicted action.
        """
        return self.forward(state).detach().numpy()