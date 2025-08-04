from skeleton import Environment

import matplotlib.pyplot as plt
import gymnasium as gym


class Ant(Environment):
    # render_mode = human, rgb_array, or depth_array
    def __init__(self, image_states=False, render_mode='rgb_array', width=400, height=400):
        super().__init__()
        self._env = gym.make("Ant-v5", render_mode=render_mode, width=width, height=height)
        self.state_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.current_state = None
        self.image_states = image_states
        self.frames = []
        self.reset()
        if self.image_states:
            self.step = self._step_image
            self.observation_dims = self.render().shape
        else:
            self.step = self._step_representation
            self.observation_dims = self._env.observation_space#.dimensions #TODO


    def get_actions(self):
        if isinstance(self._env.action_space, gym.spaces.Discrete):
            actions = list(range(self._env.action_space.n))
        else:
            actions = self._env.action_space
        return actions    

    def render(self):
        """
        Render the current state of the environment.

        :return: The rendered image as np.ndarray.
        """
        return self._env.render()
    

    
    def render_image(self):
        """
        Render the current state of the environment as an image.
        
        :return: The rendered image as np.ndarray.
        """
        frame = self.render()
        plt.figure()
        ax = plt.axes(xlim=(0, frame.shape[1]), ylim=(frame.shape[0], 0), frameon=False)
        ax.axis('off')
        ax.imshow(frame)



    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        
        :return: The initial observation of the environment.
        """
        self.current_state, _ = self._env.reset()
        return self.current_state
    
    def step(self, action):
        raise NotImplementedError(".step() should be overwritten in __init__()!")


    def _step_representation(self, action):
        """
        Take a step in the environment based on the given action.
        
        :param action: The action to take.
        :return: A tuple containing the next state, reward, done flag, 
            and additional info (s, r, done, info).
        """
        self.current_state, reward, done, truncated, info = self._env.step(action)
        return self.current_state, reward, done, truncated, info
    

    def _step_image(self, action):
        """
        Take a step in the environment based on the given action and return an image.
        """
        _, reward, done, truncated, info = self._env.step(action)
        self.current_state = self.render()
        return self.current_state, reward, done, truncated, info
    

    def record(self):
        # needs to be debugged!!!
        """
        Record a video of the environment's interaction.
        
        :param filename: The name of the file to save the video.
        """
        if not self.image_states:
            raise ValueError("Recording is only supported for image states.")
        self.frames.append(self.current_state)

        

    
