from abc import ABC, abstractmethod

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


