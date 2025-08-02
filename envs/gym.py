from skeleton import Environment

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


    

    def render(self):
        """
        Render the current state of the environment.

        :return: The rendered image as np.ndarray.
        """
        return self._env.render()

    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        
        :return: The initial observation of the environment.
        """
        self.current_state, _ = self._env.reset()
        return self.current_state
    
    def step(self, action):
        pass # gets overridden in __init__ based on image_states


    def _step_representation(self, action):
        """
        Take a step in the environment based on the given action.
        
        :param action: The action to take.
        :return: A tuple containing the next state, reward, done flag, 
            and additional info (s, r, done, info).
        """
        self.current_state, reward, done, info = self._env.step(action)
        return self.current_state, reward, done, info
    

    def _step_image(self, action):
        """
        Take a step in the environment based on the given action and return an image.
        """
        _, reward, done, info = self._env.step(action)
        self.current_state = self.render()
        return self.current_state, reward, done, info
    

    def record(self):
        """
        Record a video of the environment's interaction.
        
        :param filename: The name of the file to save the video.
        """
        if not self.image_states:
            raise ValueError("Recording is only supported for image states.")
        self.frames.append(self.current_state)

        

    
