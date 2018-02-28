# A-Reinforcement-Learning-Agent-Playing-Pacman

![pacman](https://user-images.githubusercontent.com/19307995/36691106-ee85eb2c-1b3c-11e8-886f-462adba3231c.png)


## Description
In this project I implemented a reinforcement learning agent to play pacman based on features of each **Markov** state. The environment is 
represented using **Markov Decision Processes**, and the agent acts in the environment to estimate the value of each action and based on 
that it plays and wins nearly each single game. There is a basic simulation for robot controller (Crawler), where I applied the same learning algorithms to the crawler.



## Table of contents

There are 8 main files:

+ A file contains a value iteration agent for solving known MDPs, **valueIterationAgents.py**.

+ A file contains Q-learning agents for Gridworld, Crawler and Pacman, **qlearningAgents.py**.

+ A file defines methods on general MDPs, **mdp.py**.

+ A file that defines the base classes ValueEstimationAgent and QLearningAgent, which the agents will extend, **learningAgents.py**.

+ A file contains the Gridworld implementation, **gridworld.py**.

+ A file that contains classes for extracting features on (state,action) pairs. Used for the approximate Q-learning agent (in qlearningAgents.py), **featureExtractors.py**.

+ A file that contains utilities, including util.Counter, which is particularly useful for Q-learners, **util.py**.


+ A file with different values of discounting factor and noise factor to see the effect of changing them on the behaviour of the agent,   **analysis.py**.


## Installation & Dependencies
+ All the code is written in **python 2.7**, so you might consider creating a separate environment if you'r using Python 3 on your system. To be able to create an environment head to [Anaconda](https://anaconda.org/).


## Licence

Berkeley.
