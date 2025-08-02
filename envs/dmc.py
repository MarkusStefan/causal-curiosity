from skeleton import Environment

from dm_control import suite
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML




class Walker(Environment):
    def __init__(self, domain_name='walker', task_name='walk', image_states=False):
        super().__init__()
        self._env = suite.load(domain_name='walker', task_name='walk')
        self.image_states = image_states
        self.state_space = self._env.observation_spec()
        self.action_space = self._env.action_spec()
        self.observation_space = self._env.observation_spec()
        self.current_state = None
        self.reset()


    def step(self, action):
        """
        Take a step in the environment based on the given action.
        
        :param action: The action to take.
        :return: A tuple containing the next state, reward, done flag, 
            and additional info (s, r, done, info).
        """
        time_step = self._env.step(action)
        if self.image_states:
            self.render()
        self.current_state = time_step.observation
        reward = time_step.reward
        done = time_step.last()
        info = {}
        return self.current_state, reward, done, info


    def render(self):
        """
        Render the current state of the environment.
        """
        return self._env.physics.render(camera_id='track', width=640, height=480)

