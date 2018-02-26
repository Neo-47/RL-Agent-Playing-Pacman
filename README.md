# A-Reinforcement-Learning-Agent-Playing-Pacman

![pacman](https://user-images.githubusercontent.com/19307995/36691106-ee85eb2c-1b3c-11e8-886f-462adba3231c.png)


# Description
In this project I implemented a reinforcement learning agent to play pacman based on features of each **Markov** state. The environment is 
represented using **Markov Decision Processes**, and the agent acts in the environment to estimate the value of each action and based on 
that it plays and wins nearly each single game. There is a basic simulation for robot controller (Crawler), where I applied the same learning algorithms to the crawler.


The project is one of 5 projects offered in [Berkeley's cs188 intro to AI.](http://ai.berkeley.edu/home.html)


# Table of contents

There are 3 main files:

+ A file contains a value iteration agent for solving known MDPs, **valueIterationAgents.py**.

+ A file contains Q-learning agents for Gridworld, Crawler and Pacman, **qlearningAgents.py**.

+ A file with different values of discounting factor and noise factor to see the effect of changing them of the behaviour on the agent,   **analysis.py**.


# Installation & Dependencies
+ All the code is written in **python 2.7**, so you might consider creating a separate environment if you'r using Python 3 on your system. To be able to create an environment head to [Anaconda](https://anaconda.org/).
