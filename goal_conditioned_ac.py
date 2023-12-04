from models import Actor, Critic
import torch
import torch.nn as nn
import copy

def merge_obs_goal(state):


    """
    Merge the observation and goal to create a new state.

    If both are 1D arrays, then concatenate them.

    Args:
        state (dict): The state dictionary.

    Returns:
        modified_state (torch.Tensor): The modified state = merged observation and goal.

    """

    if isinstance(state, dict) and ('observation' in state) and ('goal' in state):
        
        obs = state['observation']
        goal = state['goal']
        assert obs.shape == goal.shape

        # Concatenate the observation and goal
        assert len(obs.shape) == 2
        modified_state = torch.cat([obs, goal], dim=-1)
        assert obs.shape[0] == modified_state.shape[0]
        assert obs.shape[1] + goal.shape[1] == modified_state.shape[1]

    else:
        raise ValueError("The state should be a dictionary with 'observation' and 'goal' keys but got {}".format(state))
    
    return modified_state



class GoalConditionedActor(Actor):

    """
    Actor network for goal-conditioned RL.
    Takes in the observation and goal as input and outputs the action.
    """

    def forward(self, state):

        # Merge the observation and goal
        modified_state = merge_obs_goal(state)

        return super().forward(modified_state)
    

class GoalConditionedCritic(Critic):

    """
    Critic network for goal-conditioned RL.
    Takes in the observation, goal and action as input and outputs the Q-value.
    """


    def forward(self, state, action):

        # Merge the observation and goal
        modified_state = merge_obs_goal(state)

        return super().forward(modified_state, action)
    


class EnsembledCritic(nn.Module):

    def __init__(self, CriticInstance, ensemble_size = 3):

        super().__init__()

        self.ensemble_size = ensemble_size

        self.critics = nn.ModuleList([CriticInstance])

        for _ in range(self.ensemble_size - 1):
            critic_copy = copy.deepcopy(CriticInstance)
            critic_copy.reset_parameters()
            self.critics.append(critic_copy)


    def forward(self, *args, **kwargs):

        
        q_list = [critic(*args, **kwargs) for critic in self.critics]

        return q_list
    
    def state_dict(self):

        state_dict = {}

        for i, critic in enumerate(self.critics):
            state_dict[f'critic{i}'] = critic.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
            
            for i, critic in enumerate(self.critics):
                critic.load_state_dict(state_dict[f'critic{i}']) 
