from models import DQLAgent, QLearningAgent
from maze import Maze

def main():
    maze_size = 18
    traps = 7
    state_size = maze_size * maze_size
    action_size = 4
    episodes = 5000
    random_seed = 42

    print('Deep Q Learning Results:')
    print(f'Maze: {maze_size}', f'Traps: {traps}', f'Episodes: {episodes}')
    agent = DQLAgent(state_size, action_size)
    maze = Maze(size=maze_size, num_traps=traps, random_seed=random_seed)
    agent.train_without_visualization(maze, episodes=episodes, batch_size=64)

    print('\nQ Learning Results:')
    print(f'Maze: {maze_size}', f'Traps: {traps}', f'Episodes: {episodes}')
    maze = Maze(size=maze_size, num_traps=traps, random_seed=random_seed)
    agent = QLearningAgent(maze)
    agent.train_q_without_visualization(episodes=episodes)

if __name__ == "__main__":
    main()