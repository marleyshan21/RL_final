# from pud.dependencies import *
import os
import torch
import numpy as np
from addl_utils import set_global_seed, set_env_seed, AttrDict
import argparse
from nav2d_helper.envs import env_load_fn
from goal_conditioned_agent import UVFDDPG
from replay_buffer import ReplayBuffer
from policy import GaussianPolicy
from policy import SearchPolicy
from visualize import visualize_compare_search
from runner import train_eval, eval_pointenv_dists
from collector import Collector

def main(cfg_file, train=False):

    
    cfg = AttrDict(**eval(open(cfg_file, 'r').read()))
    print(cfg)

    # Set the random seed
    set_global_seed(cfg.seed)

    # Create the train environment
    env = env_load_fn(cfg.env.env_name, cfg.env.max_episode_steps,
                        resize_factor=cfg.env.resize_factor,
                        terminate_on_timeout=False)
    # Set the environment seed
    set_env_seed(env, cfg.seed + 1)

    # Create eval environment
    eval_env = env_load_fn(cfg.env.env_name, cfg.env.max_episode_steps,
                        resize_factor=cfg.env.resize_factor,
                        terminate_on_timeout=True)
    # Set the environment seed
    set_env_seed(eval_env, cfg.seed + 2)

    obs_dim = env.observation_space['observation'].shape[0]
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f'obs dim: {obs_dim}, goal dim: {goal_dim}, state dim: {state_dim}, action dim: {action_dim}, max action: {max_action}')

    # Create the agent
    agent = UVFDDPG(
        state_dim,
        action_dim,
        max_action,
        **cfg.agent,
    )
    print(agent)

    # Create the replay buffer
    replay_buffer = ReplayBuffer(obs_dim, goal_dim, action_dim, **cfg.replay_buffer)

    if train:
        # Create the policy
        policy = GaussianPolicy(agent)

        
        train_eval(policy,
                agent,
                replay_buffer,
                env,
                eval_env,
                eval_func=eval_pointenv_dists,
                **cfg.runner,
                )
        
        # Save the model
        torch.save(agent.state_dict(), os.path.join(cfg.ckpt_dir + 'agent_' +  cfg.env.env_name + '_' + str(cfg.runner.num_iterations) + '.pth'))
    
    else:
        
        # This is to evaluate the agent

        # Load the model
        ckpt_file = os.path.join(cfg.ckpt_dir, 'actor.pth')
        agent.load_state_dict(torch.load(ckpt_file))
        agent.eval()

        from visualize import visualize_trajectory
        eval_env.duration = 100 # We'll give the agent lots of time to try to find the goal.
        visualize_trajectory(agent, eval_env, difficulty=0.5)


        env.set_sample_goal_args(prob_constraint=0.0, min_dist=0, max_dist=np.inf)
        rb_vec = Collector.sample_initial_states(eval_env, replay_buffer.max_size)
        pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)
        
        
        search_policy = SearchPolicy(agent, rb_vec, pdist=pdist, open_loop=True)
        eval_env.duration = 300 # We'll give the agent lots of time to try to find the goal.

        
        # Plot the search path found by the search policy
        visualize_compare_search(agent, search_policy, eval_env, difficulty=0.2)

        print('Done')



if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)  
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()

    train = args.train  
    cfg_file = args.config_file

    main(cfg_file, train=train)

