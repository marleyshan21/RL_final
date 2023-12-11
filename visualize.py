
from collector import Collector
from nav2d_helper.envs import plot_walls, set_env_difficulty
from addl_utils import set_global_seed, set_env_seed
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_trajectory(agent, eval_env, difficulty=0.5):
    set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(8, 4))
    for col_index in range(2):
        plt.subplot(1, 2, col_index + 1)
        plot_walls(eval_env.walls)
        goal, observations_list, _, _ = Collector.get_trajectory(agent, eval_env)
        obs_vec = np.array(observations_list)

        print(f'traj {col_index}, num steps: {len(obs_vec)}')

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                    color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                    color='green', s=200, label='end')
        plt.scatter([goal[0]], [goal[1]], marker='*',
                    color='green', s=200, label='goal')
        if col_index == 0:
            plt.legend(loc='lower left', bbox_to_anchor=(0.3, 1), ncol=3, fontsize=16)
    plt.show()


def visualize_buffer(rb_vec, eval_env):
    plt.figure(figsize=(6, 6))
    plt.scatter(*rb_vec.T)
    plot_walls(eval_env.walls)
    plt.show()


def visualize_pairwise_dists(pdist):
    plt.figure(figsize=(6, 3))
    plt.hist(pdist.flatten(), bins=range(20))
    plt.xlabel('predicted distance')
    plt.ylabel('number of (s, g) pairs')
    plt.show()


def visualize_graph(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8):
    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)
    pdist_combined = np.max(pdist, axis=0)
    plt.scatter(*rb_vec.T)
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff:
                s_j = rb_vec[j]
                plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)
    plt.show()


def visualize_graph_ensemble(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8):
    ensemble_size = pdist.shape[0]
    plt.figure(figsize=(5 * ensemble_size, 4))
    for col_index in range(ensemble_size):
        plt.subplot(1, ensemble_size, col_index + 1)
        plot_walls(eval_env.walls)
        plt.title('critic %d' % (col_index + 1))

        plt.scatter(*rb_vec.T)
        for i, s_i in enumerate(rb_vec):
            for count, j in enumerate(np.argsort(pdist[col_index, i])):
                if count < edges_to_display and pdist[col_index, i, j] < cutoff:
                    s_j = rb_vec[j]
                    plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)
    plt.show()


def visualize_full_graph(g, rb_vec, eval_env):
    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)
    plt.scatter(rb_vec[g.nodes, 0], rb_vec[g.nodes, 1])

    edges_to_plot = g.edges
    edges_to_plot = np.array(list(edges_to_plot))

    for i, j in edges_to_plot:
        s_i = rb_vec[i]
        s_j = rb_vec[j]
        plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)

    plt.title(f'|V|={g.number_of_nodes()}, |E|={len(edges_to_plot)}')
    plt.show()


def visualize_search_path(search_policy, eval_env, difficulty=0.5):
    set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:
        state = eval_env.reset()
        start = state['observation']
        goal = state['goal']

        search_policy.select_action(state)
        waypoints = search_policy.get_waypoints()
    else:
        goal, observations, waypoints, _ = Collector.get_trajectory(search_policy, eval_env)
        start = observations[0]

    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)

    waypoint_vec = np.array(waypoints)

    print(f'waypoints: {waypoint_vec}')
    print(f'waypoints shape: {waypoint_vec.shape}')
    print(f'start: {start}')
    print(f'goal: {goal}')

    plt.scatter([start[0]], [start[1]], marker='+',
                color='red', s=200, label='start')
    plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')
    plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
    plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=16)
    plt.show()


def visualize_compare_search(agent, search_policy, eval_env, difficulty=0.5, seed=0):
    set_env_difficulty(eval_env, difficulty)

    

    plt.figure(figsize=(12, 5))
    for col_index in range(2):
        title = 'no_search' if col_index == 0 else 'search'
        plt.subplot(1, 2, col_index + 1)
        plot_walls(eval_env.walls)
        use_search = (col_index == 1)

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        if use_search:
            policy = search_policy
        else:
            policy = agent
        goal, observations, waypoints, _ = Collector.get_trajectory(policy, eval_env)
        start = observations[0]

        # save the start, goal, observations, waypoints in a csv file
        # with open('./results/info_' + str(eval_env.env_name) + '_' + str(difficulty) + str(title) + str(seed) + '.csv', 'w') as f:
        #     f.write('start_x, start_y, goal_x, goal_y\n')
        #     f.write(str(start[0]) + ', ' + str(start[1]) + ', ' + str(goal[0]) + ', ' + str(goal[1]) + '\n')
        #     f.write('observation_x, observation_y\n')
        #     for obs in observations:
        #         f.write(str(obs[0]) + ', ' + str(obs[1]) + '\n')
        #     f.write('waypoint_x, waypoint_y\n')
        #     for waypoint in waypoints:
        #         f.write(str(waypoint[0]) + ', ' + str(waypoint[1]) + '\n')


        obs_vec = np.array(observations)
        waypoint_vec = np.array(waypoints)

        print(f'policy: {title}')
        print(f'start: {start}')
        print(f'goal: {goal}')
        print(f'steps: {obs_vec.shape[0] - 1}')
        print('-' * 10)

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
        plt.scatter([start[0]], [start[1]], marker='+',
                    color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                    color='green', s=200, label='end')
        plt.scatter([goal[0]], [goal[1]], marker='*',
                    color='green', s=200, label='goal')
        plt.title(title, fontsize=24)

        if use_search:
            
            waypoint_0 = waypoints[0]
            plt.scatter([waypoint_0[0]], [waypoint_0[1]], marker='s',
                        color='darkkhaki', s=20, label='waypoint')

            for waypoint in waypoints[1:]:
                plt.scatter([waypoint[0]], [waypoint[1]], marker='s',
                            color='darkkhaki', s=20)

            # plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
            plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
    plt.show()


def plot_policy_outputs(observations, waypoints, goal, eval_env):
    start = observations[0]

    obs_vec = np.array(observations)
    waypoint_vec = np.array(waypoints)

    # print(f'waypoints: {waypoint_vec}')
    print(f'waypoints shape: {waypoint_vec.shape}')

    print(f'start: {start}')
    print(f'goal: {goal}')
    print(f'steps: {obs_vec.shape[0] - 1}')
    print('-' * 10)
    plot_walls(eval_env.walls)
    plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
    plt.scatter([start[0]], [start[1]], marker='+',
                color='red', s=200, label='start')
    # plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
    #             color='green', s=200, label='end')
    plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')
    plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
    plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
    plt.show()

def plot_points(points, eval_env):

    # points is a list of points

    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)

    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
    for i, point in enumerate(points):

        plt.scatter([point[0]], [point[1]], marker='+',
                    color=colors[i], s=200)
        
    plt.show()



def save_policy_outputs(observations, waypoints, goal, eval_env, count, policy_name, difficulty=0.5):


    start = observations[0]

    obs_vec = np.array(observations)
    waypoint_vec = np.array(waypoints)

    # 0.401111323038737, 0.716620839436849, 0.5051129659016927, 0.22775245666503907
    start = [0.401111323038737, 0.716620839436849]
    goal = [0.5051129659016927, 0.22775245666503907]

    waypoint_vec = [
         [0.401111323038737, 0.716620839436849], 
        [0.41654980977376305, 0.8144593302408855], [0.44756505330403645, 0.8797556559244791],
                    
                    [0.4077930196126302, 0.906159159342448], [0.3028385670979818, 0.8644063313802083],

                    [0.29086931864420573, 0.8029021708170573], [0.2314434305826823, 0.7004107666015625],
                    [0.22064112345377604, 0.6474564107259114], [0.23370496114095052, 0.5956202189127604],

                    [0.26858904520670573, 0.5519497680664063], [0.2699370320638021, 0.4960146077473958],

                    [0.29184059143066404, 0.4494481404622396], [0.28262110392252604, 0.4049163309733073],

                    [0.277876459757487, 0.3468177541097005], [0.2581261189778646, 0.2963064575195313],

                    [0.2792675272623698, 0.24534866333007813], [0.2806652323404948, 0.1914931869506836],
                    [0.31513173421223956, 0.1590250523885091], [0.35916392008463544, 0.11911465962727864],
                    [0.4088574981689453, 0.1180446751912435], [0.46737904866536456, 0.16967459360758463],
                    [0.5051129659016927, 0.22775245666503907]
    ]            


    print(f'start: {start}')
    print(f'goal: {goal}')
    print(f'steps: {obs_vec.shape[0] - 1}')
    print('-' * 10)
    plot_walls(eval_env.walls)
    # plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
    plt.scatter([start[0]], [start[1]], marker='+',
                color='red', s=200, label='start')
    # plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
    #             color='green', s=200, label='end')
    plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')
    
    for waypoint in waypoint_vec:
        plt.plot([waypoint[0]], [waypoint[1]], 'y-s', alpha=0.3)

    # plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
    # plt.legend(loc='lower left', ncol=4, fontsize=10)

    # create a directory to store the plots
    if not os.path.exists('./results/plots_' + eval_env.env_name + '/' + str(difficulty)):
        os.makedirs('./results/plots_' + eval_env.env_name + '/' + str(difficulty))
    

    if policy_name == 'baseline_goal_conditioned':
        name = 'baseline'
    else:
        name = 'search'

    plt.savefig('./results/plots_' + eval_env.env_name + '/' + str(difficulty) + '/' + name + '_' + str(count) + '.png')

    plt.show()
    plt.close()