import numpy as np


class Collector:
    def __init__(self, policy, buffer, env, initial_collect_steps=0):
        self.buffer = buffer
        self.env = env
        self.policy = policy

        self.steps = 0
        self.state = env.reset()
        self.initial_collect_steps = initial_collect_steps

    def step(self, num_steps):

        """
        Method to collect data from the environment using the policy.
        """

        for _ in range(num_steps):
            if self.steps < self.initial_collect_steps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(self.state)
            # print("action is", action)

            next_state, reward, done, info = self.env.step(np.copy(action))

            # Add the transition to the replay buffer and update the state
            if info.get('last_timestep', False):
                self.buffer.add(self.state, action, info['terminal_observation'], reward, done)
                self.state = next_state
            else:
                self.buffer.add(self.state, action, next_state, reward, done)
                self.state = next_state

            self.steps += 1

    @classmethod
    def sample_initial_states(cls, eval_env, num_states):

        """
        Sample `num_states` initial states from the environment.
        """

        rb_vec = []
        for _ in range(num_states):
            rb_vec.append(eval_env.reset())
        rb_vec = np.array([x['observation'] for x in rb_vec])
        return rb_vec

    @classmethod
    def eval_agent(cls, policy, eval_env, n, by_episode=True):
        """
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """

        c = 0
        r = 0
        rewards = []
        state = eval_env.reset()
        while c < n:
            action = policy.select_action(state)
            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode: c += 1

            r += reward
            if done:
                rewards.append(r)
                if by_episode: c += 1
                r = 0
        return rewards

    @classmethod
    def step_cleanup(cls, search_policy, eval_env, num_steps):
        c = 0
        while c < num_steps:
            goal = search_policy.get_goal_in_rb()
            state = eval_env.reset()
            done = False

            while True:
                state['goal'] = goal
                try:
                    action = search_policy.select_action(state)
                except Exception as e:
                    raise e

                state, reward, done, info = eval_env.step(np.copy(action))
                c += 1

                if done or c >= num_steps or search_policy.reached_final_waypoint:
                    break

    @classmethod
    def get_trajectory(cls, policy, eval_env):

        """
        Method to collect a trajectory from the environment using the policy.
        
        """


        ep_observation_list = []
        ep_waypoint_list = []
        ep_reward_list = []

        state = eval_env.reset()
        ep_goal = state['goal']

        while True:
            ep_observation_list.append(state['observation'])
            action = policy.select_action(state) # NOTE: state['goal'] may be modified
            # print("action is", action)
            ep_waypoint_list.append(state['goal'])
            state, reward, done, info = eval_env.step(np.copy(action))

            ep_reward_list.append(reward)
            if done:
                ep_observation_list.append(info['terminal_observation']['observation'])
                break

        return ep_goal, ep_observation_list, ep_waypoint_list, ep_reward_list
    


    @classmethod
    def get_trajectory_given(cls, policy, eval_env, start, goal):

        """
        Method to collect a trajectory from the environment using the policy
        given a goal and start state.
        
        """


        ep_observation_list = []
        ep_waypoint_list = []
        ep_reward_list = []

        state = eval_env.reset()

        print("originally, state is", state)    

        state['observation'] = np.array(start)
        state['goal'] = np.array(goal)

        ep_goal = state['goal']

        print("state is", state)


        while True:
            ep_observation_list.append(state['observation'])
            action = policy.select_action(state) # NOTE: state['goal'] may be modified
            ep_waypoint_list.append(state['goal'])
            state, reward, done, info = eval_env.step(np.copy(action))

            ep_reward_list.append(reward)
            if done:
                ep_observation_list.append(info['terminal_observation']['observation'])
                break

        return ep_goal, ep_observation_list, ep_waypoint_list, ep_reward_list