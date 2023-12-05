from ddpg import DDPG
from models import Actor, Critic
from goal_conditioned_ac import GoalConditionedActor, GoalConditionedCritic, EnsembledCritic
import functools    
import copy
import torch
import torch.nn.functional as F
import numpy as np

class UVFDDPG(DDPG):


    def __init__(self, *args, discount = 1,
                 num_bins = 1, use_distributional_rl = False,
                 ensemble_size = 1, CriticCls = GoalConditionedCritic,
                 **kwargs):
        
        self.num_bins = num_bins
        self.use_distributional_rl = use_distributional_rl
        self.ensemble_size = ensemble_size

        if self.use_distributional_rl:
            CriticCls = functools.partial(CriticCls, output_dim = self.num_bins)
            assert discount == 1, "Discount factor must be 1 for distributional RL"

        super().__init__(*args, discount = discount, ActorCls=GoalConditionedActor,
                         CriticCls = CriticCls, **kwargs)
        

        if self.ensemble_size > 1:
            self.critic = EnsembledCritic(self.critic, ensemble_size = self.ensemble_size)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.load_state_dict(self.critic.state_dict())


            for i in range(1, len(self.critic.critics)):
                critic_copy = self.critic.critics[i]
                self.critic_optimizer.add_param_group({'params': critic_copy.parameters()})




    def select_action(self, state):
        
        """
        This method is used to select an action given the state
        using the actor network.
        
        """


        with torch.no_grad():

            
            state = dict(
                observation = torch.FloatTensor(state['observation'].reshape(1, -1)),
                goal = torch.FloatTensor(state['goal'].reshape(1, -1))
            )
            
            return self.actor(state).cpu().detach().numpy().flatten()
        


    def get_q_values(self, state, aggregate='mean'):

        """
        This method is used to get the Q-values given the state and action
        using the critic network.
        
        This handles the case where the critic is an ensemble of critics 
        and processes the output of each critic to get the Q-values before
        aggregating them and using them to compute the loss.
        """


        q_values = super().get_q_values(state)

        if not isinstance(q_values, list):
            q_values_list = [q_values]
        else:
            q_values_list = q_values


        expected_q_values_list = []

        if self.use_distributional_rl:
            """
            If flag is set to True, then we use distributional RL.
            Here we convert the Q-values to a probability distribution
            over the bins and then compute the expected Q-values.
            """

            for q_values in q_values_list:
                q_probs = F.softmax(q_values, dim =1)
                batch_size = q_probs.shape[0]

                neg_bin_range = -torch.arange(1, self.num_bins + 1, dtype = torch.float)
                tiled_bin_range = neg_bin_range.unsqueeze(0).repeat(batch_size, 1)
                assert q_probs.shape == tiled_bin_range.shape

                expected_q_values = torch.sum(q_probs * tiled_bin_range, dim = 1, keepdim = True)
                expected_q_values_list.append(expected_q_values)

        else:

            """
            If flag is set to False, then we use regular RL
            Here we compute the expected Q-values directly using the Q-values
            from each critic.
            """

            expected_q_values_list = q_values_list



        expected_q_values = torch.stack(expected_q_values_list)

        if aggregate is not None:
            if aggregate == 'mean':
                expected_q_values = torch.mean(expected_q_values, dim = 0)
            elif aggregate == 'min':
                expected_q_values = torch.min(expected_q_values, dim = 0)[0]
            else:
                raise ValueError("Invalid aggregate function {}".format(aggregate))
            

        if not self.use_distributional_rl:
            """
            Clip the Q-values to be in the range [-num_bins, 0]
            """

            min_q_value = -1.0 * self.num_bins
            max_q_value = 0.0
            expected_q_values = torch.clamp(expected_q_values, min_q_value, max_q_value)

        return expected_q_values
    


    def critic_loss(self, current_q, target_q, reward, done):
        
        if not isinstance(current_q, list):
            current_q_list = [current_q]
            target_q_list = [target_q]
        else:
            current_q_list = current_q
            target_q_list = target_q

        critic_loss_list = []

        for current_q, target_q in zip(current_q_list, target_q_list):

            if self.use_distributional_rl:

                #  convert the Q-values to a probability distribution over the bins using softmax
                target_q_probs = F.softmax(target_q, dim = 1)
                batch_size = target_q_probs.shape[0]

                #  create a one-hot vector for the terminal state
                one_hot = torch.zeros(batch_size, self.num_bins)
                one_hot[: , 0] = 1.0

                #  Split the Q-values into three parts - first, middle and last bins
                col_1 = torch.zeros((batch_size, 1))
                col_middle = target_q_probs[:, :-2]
                col_last = torch.sum(target_q_probs[:, -2:], dim = 1, keepdim = True)

                shifted_target_q_probs = torch.cat([col_1, col_middle, col_last], dim = 1)
                assert shifted_target_q_probs.shape == one_hot.shape

                # Compute the TD-targets 
                # If the state is terminal, then the TD-target is the one-hot vector
                # Else, the TD-target is the shifted Q-values
                td_targets = torch.where(done.bool(), one_hot, shifted_target_q_probs).detach()

                # Compute the critic loss using the TD-targets and the current Q-values
                critic_loss = torch.mean(-torch.sum(td_targets * torch.log_softmax(current_q, dim = 1), dim = 1))


            else:

                critic_loss = super().critic_loss(current_q, target_q, reward, done)

            critic_loss_list.append(critic_loss)

        critic_loss = torch.mean(torch.stack(critic_loss_list))

        return critic_loss




    def get_dist_to_goal(self, state, **kwargs):

        """
        This method is uses the learned Q-values to compute the distance to the goal.
        
        """

        
        with torch.no_grad():

            state = dict(
                observation = torch.FloatTensor(state['observation']),
                goal = torch.FloatTensor(state['goal'])
            )
            q_values = self.get_q_values(state, **kwargs)
            return -1.0 * q_values.cpu().detach().numpy().squeeze(-1)
        



    def get_pairwise_dist(self, obs_vec, goal_vec = None, aggregate='mean', max_search_steps=7, masked=False):


        if goal_vec is None:
            goal_vec = obs_vec

        dist_matrix = []

        for obs_index in range(len(obs_vec)):
            obs = obs_vec[obs_index]
            obs_repeat_tensor = np.repeat([obs], len(goal_vec), axis=0)
            state = {'observation': obs_repeat_tensor, 'goal': goal_vec}
            dist = self.get_dist_to_goal(state, aggregate=aggregate)
            dist_matrix.append(dist)

        pairwise_dist = np.stack(dist_matrix)

        if aggregate is None:
            pairwise_dist = np.transpose(pairwise_dist, [1, 0, 2])

        if masked:
            mask = (pairwise_dist > max_search_steps)
            return  np.where(mask, np.full(pairwise_dist.shape, np.inf), pairwise_dist)
        
        else:
            return pairwise_dist