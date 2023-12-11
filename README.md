# Search on the Replay Buffer

[**Project Report**]() | [**Project Slides**]()
## SoRB Algorithm - Introduction


An intelligent agent will be able to solve complex,
temporally extended tasks. A lot of research has been done
in the fields of Path planning and Reinforcement Learning in
order to make agents intelligent and be able to learn tasks
of varying horizons. 

Classical Path planning algorithms like
A* and RRT* can find the shortest path and reason over long
horizons if provided with a local policy and an understanding of
the distance metric. This becomes a roadblock when we reason
over high-dimensional observation spaces like image-based
tasks. A set of complex RL problems called Goal Conditioned
Reinforcement Learning helps train an agent to learn a policy
to achieve different goals under particular scenarios. These
algorithms excel in learning these policies and can handle high-
dimensional observations. However, these methods suffer from
failure to handle long-horizon tasks.

A recent approach called Search on the Replay Buffer
(SORB) aims to utilize the strengths of both Path planning
and Goal Conditioned Reinforcement Learning algorithms. The
approach divides the long-horizon tasks into a series of easier
sub-goals in order to accomplish the main task.
In this work, we have written and tested the algorithm in
R2 space for different motion planning scenarios to build an
intuition behind the method.

## Installation

```bash
git clone https://github.com/marleyshan21/RL_final.git
cd RL-final
pip install -r requirements.txt
```

## Usage and Training

The parameters such as the number of iterations, option to use distribution RL, the desired environment can be changed by altering the `configs/config_PointEnv.py` file.

For training the agent on a particular environment set in the config file, run the following command.

```bash
python main.py configs/config_PointEnv.py --train
```

For evaluation, use 

```bash
python main.py configs/config_PointEnv.py
```

## Collaboration

Done in collaboration with - Hardik Devrangadi
