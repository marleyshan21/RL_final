from addl_utils import *
import torch.nn as nn
import torch.nn.functional as F
import copy
from models import Actor, Critic

class DDPG(nn.Module):

    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, 
                 actor_update_interval=1,
                 targets_update_interval=1,
                    tau=0.005,
                    ActorCls=Actor,
                    CriticCls=Critic):
        super().__init__()

        # Save the parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.actor_update_interval = actor_update_interval
        self.targets_update_interval = targets_update_interval
        self.tau = tau

        # Create the actor and critic networks
        self.actor = ActorCls(state_dim, action_dim, max_action)

        # using a separate target actor network 
        # improves the stability of the algorithm
        self.actor_target =  copy.deepcopy(self.actor)

        #  initialize the target actor network weights to be the same as the actor network
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-07)

        # Create the critic networks
        self.critic = CriticCls(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, eps=1e-07)

        self.optimize_iterations = 0



    def select_action(self, state):

        # No gradient calculation is needed
        with torch.no_grad():

            # Convert the state to a tensor
            state = torch.FloatTensor(state.reshape(1, -1))

            # Get the action from the actor network given the state
            action = self.actor(state).cpu().detach().numpy().flatten()
            return action
    


    def get_q_values(self, state):

        #  get actions predicted by the actor network 
        #  for the given state
        actions = self.actor(state)

        #  get the Q-values predicted by the critic network
        # for the given state and action pairs
        q_values = self.critic(state, actions)

        return q_values
    

    def critic_loss(self, current_q, target_q, reward, done):

        td_targets = reward + ((1 - done) * self.discount * target_q).detach()
        critic_loss = F.mse_loss(current_q, td_targets)

        return critic_loss
    

    def update_actor_target(self):

        # Update the target actor network parameters 
        # by performing a soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    def update_critic_target(self):

        # Update the target critic network parameters 
        # by performing a soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    def optimize(self, replay_buffer, iterations=1, batch_size=128):

        opt_info = dict(actor_loss=[], critic_loss=[])

        for _ in range(iterations):

            self.optimize_iterations += 1

            # Sample a batch of transitions from the replay buffer
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)

            current_q = self.critic(state, action)
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            critic_loss = self.critic_loss(current_q, target_q, reward, done)

            # Optimize the critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            opt_info['critic_loss'].append(critic_loss.cpu().detach().numpy())

            # Delayed policy updates
            if self.optimize_iterations % self.actor_update_interval == 0:

                # Compute actor loss
                actor_loss = -self.get_q_values(state).mean()

                # Optimize the actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                opt_info['actor_loss'].append(actor_loss.cpu().detach().numpy())

            # Update the target networks
            if self.optimize_iterations % self.targets_update_interval == 0:
                self.update_actor_target()
                self.update_critic_target()


        return opt_info
    


    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic=self.critic.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            optimize_iterations=self.optimize_iterations,
        )
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.optimize_iterations = state_dict['optimize_iterations']


        