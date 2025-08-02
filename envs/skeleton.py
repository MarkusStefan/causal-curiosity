from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract base class for environments.
    """

    def __init__(self):
        self.state_space = None
        self.action_space = None
        self.observation_dim = None
        self.observation_space = None
        self.current_state = None
        self.image_states = None



    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment based on the given action.
        
        :param action: The action to take.
        :return: A tuple containing the next state, reward, done flag, 
            and additional info (s, r, done, info).
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        
        :return: The initial observation of the environment.
        """
        pass



    @abstractmethod
    def render(self):
        """
        Render the current state of the environment.
        """
        pass

    @abstractmethod
    def get_actions(self):
        """
        Get the available actions in the environment.
        
        :return: A list of available actions.
        """
        pass


    # @abstractmethod
    def _get_reward(self, state, action):
        """
        Calculate the reward for a given state and action.
        This is the environment's internal reward function.

            r(s, a)
        
        :param state: The current state of the environment.
        :param action: The action taken in the environment.
        :return: The reward for the given state and action.
        """
        pass

    # @abstractmethod
    def _get_next_state(self, state, action):
        """
        Get the next state based on the current state and action.
        This is the environment's internal transition function.

            p(s' | s, a)
        
        :param state: The current state of the environment.
        :param action: The action taken in the environment.
        :return: The next state after taking the action.
        """



class MDP(Environment):
    pass

class POMDP(Environment):
    pass