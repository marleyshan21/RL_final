from collector import Collector
from nav2d_helper.envs import set_env_difficulty
import numpy as np
import time
import matplotlib.pyplot as plt
import csv

def rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    data= np.array(data)
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

def train_eval(
    policy,
    agent,
    replay_buffer,
    env,
    eval_env,
    num_iterations=int(1e6),
    initial_collect_steps=1000,
    collect_steps=1,
    opt_steps=1,
    batch_size_opt=64,
    eval_func=lambda agent, eval_env: None,
    num_eval_episodes=10,
    opt_log_interval=100,
    eval_interval=10000,
):
    collector = Collector(policy, replay_buffer, env, initial_collect_steps=initial_collect_steps)
    collector.step(collector.initial_collect_steps)
    episode_returns = []
    is_distributional_rl = agent.use_distributional_rl
    for i in range(1, num_iterations + 1):
        collector.step(collect_steps)
        agent.train()
        opt_info = agent.optimize(replay_buffer, iterations=opt_steps, batch_size=batch_size_opt)

        if i % opt_log_interval == 0:
            print(f'iteration = {i}, opt_info = {opt_info}')

        if i % eval_interval == 0:
            agent.eval()
            print(f'evaluating iteration = {i}')
            eval_func(agent, eval_env)

            returns = Collector.eval_agent(agent, eval_env, num_eval_episodes)
            episode_returns.extend(returns)
            
            print('-' * 10)
    
    # Write episode returns to a CSV file
    if is_distributional_rl == True:
        with open('./data/distributional_rl.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Return'])
            for episode, ret in enumerate(episode_returns):
                writer.writerow([episode + 1, ret])
    else: 
        with open('./data/no_distributional_rl.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Return'])
            for episode, ret in enumerate(episode_returns):
                writer.writerow([episode + 1, ret])

    plt.figure(1)
    plt.title('DDPG Training Returns vs Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(episode_returns, color='grey', label='episode returns')
    plt.plot(rolling_average(episode_returns, window_size=100), 'b', label='rolling average')
    plt.legend(loc='upper right')
    print("env:" ,eval_env.env_name)
    plt.savefig('./rewards/rewardsplot.png')
    plt.show()

def eval_pointenv_dists(agent, eval_env, num_evals=10, eval_distances=[2, 5, 10]):

    """
    This function evaluates the agent on the given environment.
    We have certain known goal distances, and we evaluate the agent's
    ability to predict these distances.
    """


    for dist in eval_distances:
        eval_env.set_sample_goal_args(prob_constraint=1, min_dist=dist, max_dist=dist) # NOTE: samples goal distances in [min_dist, max_dist] closed interval
        returns = Collector.eval_agent(agent, eval_env, num_evals)
        # For debugging, it's helpful to check the predicted distances for
        # goals of known distance.
        states = dict(observation=[], goal=[])
        for _ in range(num_evals):
            state = eval_env.reset()
            states['observation'].append(state['observation'])
            states['goal'].append(state['goal'])
        pred_dist = list(agent.get_dist_to_goal(states))

        print(f'\tset goal dist = {dist}')
        print(f'\t\treturns = {returns}')
        print(f'\t\tpredicted_dists = {pred_dist}')
        print(f'\t\taverage return = {np.mean(returns)}')
        print(f'\t\taverage predicted_dist = {np.mean(pred_dist):.1f} ({np.std(pred_dist):.2f})')


def eval_search_policy(search_policy, eval_env, num_evals=10):

    """
    Method to evaluate the search policy on the given environment.
    Returns the success rate and the time taken for evaluation.
    """


    eval_start = time.perf_counter()

    successes = 0.
    for _ in range(num_evals):
        try:
            _, _, _, ep_reward_list = Collector.get_trajectory(search_policy, eval_env)
            successes += int(len(ep_reward_list) < eval_env.duration)
        except:
            pass

    eval_end = time.perf_counter()
    eval_time = eval_end - eval_start
    success_rate = successes / num_evals
    return success_rate, eval_time