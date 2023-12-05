dict(
    seed=0,
    env=dict(
        env_name='Cross', # Check the "Walls" in envs.py for other options
        max_episode_steps=20,
        resize_factor=5, # Inflate the environment to increase the difficulty.
    ),
    runner=dict(
        num_iterations=10000,
        initial_collect_steps=1000, # Number of steps to collect at the start of training
        collect_steps=2, # Number of steps to collect per iteration
        opt_steps=1, # Number of optimization steps per iteration
        batch_size_opt=64,  # Number of transitions to sample from the replay buffer
        num_eval_episodes=10, # Evaluate the agent every 'eval_interval' iterations
        eval_interval=1000, # Evaluate the agent every 'eval_interval' iterations
    ),
    agent=dict(
        discount=1,
        num_bins=20, # equal to max_episode_steps
        use_distributional_rl=True, # Set to True to use distributional RL
        ensemble_size=3, # Number of critics to use
        targets_update_interval=5, # Update the target networks every 'targets_update_interval' iterations
        tau=0.05, # Soft update coefficient for the target networks in the actor-critic algorithm
    ),
    replay_buffer=dict(
        max_size=1000,
    ),
    ckpt_dir='./workdirs/uvfddpg_FourRooms/',
)