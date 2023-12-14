import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import pygame
import time

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class DQL(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQL, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size))

    def forward(self, x):
        return self.net(x)

class DQLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.9, epsilon= 1.0, epsilon_decay=0.995, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.model = DQL(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state).detach().numpy()
        return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        states_np = np.array([item[0] for item in minibatch])
        actions_np = np.array([item[1] for item in minibatch])
        rewards_np = np.array([item[2] for item in minibatch])
        next_states_np = np.array([item[3] for item in minibatch])
        dones_np = np.array([item[4] for item in minibatch])

        states_tensor = torch.FloatTensor(states_np).view(batch_size, -1)
        next_states_tensor = torch.FloatTensor(next_states_np).view(batch_size, -1)
        actions_tensor = torch.LongTensor(actions_np)
        rewards_tensor = torch.FloatTensor(rewards_np)
        dones_tensor = torch.FloatTensor(dones_np)

        current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_max_q_values = self.model(next_states_tensor).detach().max(1)[0]
        expected_q_values = rewards_tensor + self.gamma * next_max_q_values * (1 - dones_tensor)

        loss = nn.MSELoss()(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay

    def step(self, maze, action):
        x, y = maze.start
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < maze.size - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < maze.size - 1:
            x += 1

        maze.start = (x, y)
        next_state = maze.get_state()
        reward = -1 if maze.is_trap(maze.start) else (1 if maze.is_goal(maze.start) else -0.01)
        done = maze.is_goal(maze.start) or maze.is_trap(maze.start)
        return next_state, reward, done

    def train(self, maze, episodes, batch_size, display_size=600, sleep_time=0.0000001):
        pygame.init()
        win = pygame.display.set_mode((display_size, display_size))
        cell_size = display_size // maze.size

        success_count = 0
        win_step_counts = 0

        for episode in range(episodes):
            state = maze.reset()
            done = False
            steps = 0

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                action = self.act(state)
                next_state, reward, done = self.step(maze, action)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                self.replay(batch_size)
                steps += 1

                self.draw_maze(maze, win, cell_size)

                if maze.is_goal(maze.start):
                    success_count += 1
                    win_step_counts += steps

                time.sleep(sleep_time)

        pygame.quit()

        average_win_steps = win_step_counts / success_count if success_count else 0
        success_rate = success_count / episodes
        print(f"Success Rate: {success_rate}")
        print(f"Average Steps per Episode: {average_win_steps}")
        print(f"Wins: {success_count}, Episodes: {episodes}")

    def draw_maze(self, maze, win, cell_size):
        win.fill((0, 0, 0))
        for y in range(maze.size):
            for x in range(maze.size):
                rect = (x * cell_size, y * cell_size, cell_size, cell_size)
                if maze.grid[y, x] == -1:
                    pygame.draw.rect(win, (255, 0, 0), rect)
                elif (y, x) == maze.goal:
                    pygame.draw.rect(win, (0, 255, 0), rect)
                elif (y, x) == maze.start:
                    pygame.draw.rect(win, (0, 0, 255), rect)
                else:
                    pygame.draw.rect(win, (255, 255, 255), rect)
                pygame.draw.rect(win, (0, 0, 0), rect, 1)
        pygame.display.update()

    def train_without_visualization(self, maze, episodes, batch_size):
        success_count = 0
        win_step_counts = 0

        for episode in range(episodes):
            state = maze.reset()
            done = False
            steps = 0

            while not done:
                action = self.act(state)
                next_state, reward, done = self.step(maze, action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay(batch_size)
                steps += 1

                if maze.is_goal(maze.start):
                    success_count += 1
                    win_step_counts += steps

        average_win_steps = win_step_counts / episodes
        success_rate = success_count / episodes
        print(f"Success Rate: {success_rate}")
        print(f"Average Steps per Episode: {average_win_steps}")
        print(f"Wins: {success_count}, Episodes: {episodes}")


class QLearningAgent:
    def __init__(self, maze, alpha=0.01, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_values = {}

    def getQValue(self, state, action):
        state_key = tuple(state) if isinstance(state, np.ndarray) else state
        return self.q_values.get((state_key, action), 0.0)

    def computeValueFromQValues(self, state):
        state_key = tuple(state) if isinstance(state, np.ndarray) else state
        actions = self.getLegalActions(state_key)
        if not actions:
            return 0.0
        return max(self.getQValue(state_key, action) for action in actions)


    def computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None
        max_value = self.computeValueFromQValues(state)
        best_actions = [a for a in actions if self.getQValue(state, a) == max_value]
        return random.choice(best_actions)

    def getAction(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.getLegalActions(state))
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        state_key = tuple(state) if isinstance(state, np.ndarray) else state
        nextState_key = tuple(nextState) if isinstance(nextState, np.ndarray) else nextState

        sample = reward + self.gamma * self.computeValueFromQValues(nextState_key)
        self.q_values[(state_key, action)] = ((1 - self.alpha) * self.getQValue(state_key, action) +
                                              self.alpha * sample)

        self.epsilon *= self.epsilon_decay

    def getLegalActions(self, state):
        actions = []
        for action in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            next_state = (state[0] + action[0], state[1] + action[1])
            if 0 <= next_state[0] < self.maze.size and 0 <= next_state[1] < self.maze.size:
                actions.append(action)
        return actions

    def train_q_without_visualization(self, episodes):
        success_count = 0
        win_step_counts = 0

        for episode in range(episodes):
            state = self.maze.start
            run = True
            steps = 0

            while run:
                action = self.getAction(state)
                next_state = (state[0] + action[0], state[1] + action[1])
                reward = self.maze.get_state_o(next_state)
                self.update(state, action, next_state, reward)
                state = next_state
                steps += 1

                if self.maze.is_goal(state):
                    success_count += 1
                    win_step_counts += steps
                    break

                if self.maze.get_state_o(state) == -1:
                    break

        average_win_steps = win_step_counts / episodes
        success_rate = success_count / episodes
        print(f"Success Rate: {success_rate}")
        print(f"Average Steps per Episode: {average_win_steps}")
        print(f"Wins: {success_count}, Episodes: {episodes}")

    def train_q_with_visualization(self, episodes, display_size=600, sleep_time=0.000001):
        pygame.init()
        win = pygame.display.set_mode((display_size, display_size))
        cell_size = display_size // self.maze.size

        success_count = 0
        win_step_counts = 0

        for episode in range(episodes):
            state = self.maze.start
            run = True
            steps = 0

            while run:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                self.draw_maze(win, cell_size, state)
                pygame.display.update()

                action = self.getAction(state)
                next_state = (state[0] + action[0], state[1] + action[1])
                reward = self.maze.get_state_o(next_state)
                self.update(state, action, next_state, reward)
                state = next_state
                steps += 1

                if self.maze.is_goal(state):
                    success_count += 1
                    win_step_counts += steps
                    break

                if self.maze.get_state_o(state) == -1:
                    break

                time.sleep(sleep_time)

        pygame.quit()

        average_win_steps = win_step_counts / episodes
        success_rate = success_count / episodes
        print(f"Success Rate: {success_rate}")
        print(f"Average Steps per Episode: {average_win_steps}")
        print(f"Wins: {success_count}, Episodes: {episodes}")

    def draw_maze(self, win, cell_size, agent_pos):
        for y in range(self.maze.size):
            for x in range(self.maze.size):
                rect = (x * cell_size, y * cell_size, cell_size, cell_size)
                if (y, x) == self.maze.goal:
                    pygame.draw.rect(win, (0, 255, 0), rect)
                elif self.maze.grid[y, x] == -1:
                    pygame.draw.rect(win, (255, 0, 0), rect)
                elif (y, x) == agent_pos:
                    pygame.draw.rect(win, (0, 0, 255), rect)
                else:
                    pygame.draw.rect(win, (255, 255, 255), rect)
                pygame.draw.rect(win, (0, 0, 0), rect, 1)