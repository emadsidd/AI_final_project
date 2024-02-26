Maze Navigation with Reinforcement Learning: A Comparative Study

This project dives into the application and comparison of two reinforcement learning strategies, Q-Learning and Deep Q-Learning, aimed at solving maze navigation challenges. The core objective is to train agents to navigate through mazes of varying complexities efficiently, avoiding traps and reaching designated goals. By examining both methods in a controlled environment, this study sheds light on their respective strengths and adaptabilities to different maze scenarios.

Key Concepts:
Q-Learning: A value-based learning algorithm that operates on a discrete, finite state space, learning a function Q(s, a) which quantifies the quality of taking action 'a' in state 's'.
Deep Q-Learning (DQL): An advanced iteration of Q-Learning that uses deep neural networks to approximate the Q-function, making it suitable for tackling problems with larger and more complex state spaces.
Implementation Overview:
Maze Environment: A custom class that generates mazes as grids, with 0 representing free paths, 1 the goal, and -1 indicating traps.
DQL Components:
DQL Network: A neural network that inputs the state of the maze and outputs Q-values for each action, utilizing linear layers and ReLU activation.
DQLAgent: Manages the DQL algorithm, storing experiences, selecting actions via the epsilon-greedy policy, and updating the network based on rewards.
Q-Learning Components:
QLearningAgent: Implements the Q-Learning algorithm, maintaining and updating a table of Q-values in response to environmental interactions.
Both agents are equipped with training methods, optionally featuring visualization with Pygame to display the maze navigation process.
