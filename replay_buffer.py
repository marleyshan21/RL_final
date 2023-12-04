import numpy as np
import torch

class ReplayBuffer:


    def __init__(self, obs_dim, goal_dim, action_dim, max_size=int(1e6)):

        # Save the parameters
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Create the buffer to store the data

        # Observation buffer
        self.observation = np.zeros((max_size, obs_dim))
        # Goal buffer
        self.goal = np.zeros((max_size, goal_dim))
        # Action buffer
        self.action = np.zeros((max_size, action_dim))
        # Reward buffer
        self.reward = np.zeros((max_size, 1))
        # Next observation buffer
        self.next_observation = np.zeros((max_size, obs_dim))
        # Next goal buffer
        self.next_goal = np.zeros((max_size, goal_dim))
        # Done buffer
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):

        self.observation[self.ptr] = state['observation']
        self.goal[self.ptr] = state['goal']
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_observation[self.ptr] = next_state['observation']
        self.next_goal[self.ptr] = next_state['goal']
        self.done[self.ptr] = done

        # Update the buffer size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):

        ind = np.random.randint(0, self.size, size=batch_size)

        batch = (

            dict(
                observation=torch.FloatTensor(self.observation[ind]),
                goal=torch.FloatTensor(self.goal[ind]),
            ),
            dict(
                observation=torch.FloatTensor(self.next_observation[ind]),
                goal=torch.FloatTensor(self.next_goal[ind]),
            ),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.done[ind])
        )

        return batch