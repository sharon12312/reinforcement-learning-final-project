import gym
import highway_env
import numpy as np
import random
import os

from matplotlib import pyplot as plt
from collections import deque
from agents import DQN, DDQN, AgentFactory
from config import *

from keras.utils.vis_utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HighwayEnv:
    def __init__(self, gym_env_name=None, model=None, config_level=None, train_mode=True, load_model=False, debug=False, stochastic=False):
        if gym_env_name is None or model is None:
            print('Cannot initialize class due to missing model/gym-env-name input.')
            return

        self.env = gym.make(gym_env_name)
        self.env.configure(config_level)
        # self.env = wrap_env(self.env)

        self.model = model
        self.previous_memory = deque(maxlen=15000)

        self.debug = debug
        self.train_mode = train_mode
        self.stochastic = stochastic
        self.episodes = self.model.config['episodes'] if self.train_mode else 10

        self.frame_count = 0
        self.epsilon_greedy_frames = 10000.0
        self.update_target_network = 10000

        self.episode_reward_history = []
        self.rewards = []

        self.episode_loss_history = []
        self.losses = []

        if load_model:
            self.model.load_model()

    def play(self):
        for episode in range(self.episodes):
            if self.debug:
                print(f'\tRunning episode {episode}...')

            episode_reward = 0
            episode_loss = 0

            observation = self.env.reset()

            for iteration in range(self.model.config['iterations']):
                self.frame_count += 1

                if render:
                    self.env.render()

                action = self.model.get_action(observation, stochastic=self.stochastic, exploit=not self.train_mode)
                next_observation, reward, done, info = self.env.step(action)
                episode_reward += reward

                self.previous_memory.append([observation, action, next_observation, reward, done])
                observation = next_observation

                if self.debug:
                    print(f'iteration: {iteration}, epsilon: {self.model.epsilon:.4f}, action: {action}, reward: {reward}, done: {done}')

                if done:
                    break

                new_epsilon = self.model.epsilon - (self.model.epsilon_interval / self.epsilon_greedy_frames)
                self.model.epsilon = max(new_epsilon, self.model.min_epsilon)

                if self.train_mode and self.frame_count % 4 == 0 and len(self.previous_memory) >= self.model.config['batch_size']:
                    loss = self._train_network()
                    episode_loss = loss if episode_loss == 0 else min(episode_loss, loss)

                if self.frame_count % self.update_target_network == 0:
                    print(f'Episode: {episode}, accumulated reward: {np.mean(self.episode_reward_history):.4f}, frames count: {self.frame_count}')
                    self.model.update_prediction()

            if not self.train_mode:
                print(f'==== {type(self.model).__name__} ====')
                print(f'Reward: {episode_reward}\n')

            # store rewards history
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]
            self.rewards.append(np.mean(self.episode_reward_history))

            # store losses history
            self.episode_loss_history.append(episode_loss)
            if len(self.episode_loss_history) > 100:
                del self.episode_loss_history[:1]
            self.losses.append(np.mean(self.episode_loss_history))

    def display_video(self):
        self.env.close()
        # show_video()

    def _train_network(self):
        data = self._get_batch_data(self.model.config['batch_size'])
        loss = self.model.train(data)
        return loss

    def _get_batch_data(self, sampling_size):
        this_batch = random.sample(self.previous_memory, sampling_size)
        current_nodes, actions, next_nodes, rewards, done = list(zip(*this_batch))
        return [np.stack(current_nodes), np.array(actions), np.stack(next_nodes), np.array(rewards), np.array(done)]


def run_experiment(env_name='highway-fast-v0', agents=None, config=None, train=True, plot=False, load_model=False, stochastic=False, level='easy'):
    def plot_results(games, xlabel='', ylabel='', level='easy'):
        # initial variables
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
        name = ''

        for i, (agent_name, agent_game) in enumerate(games.items()):
            # store name for image file
            name += f'{agent_name}_'

            # fetch usage
            data = agent_game.rewards if xlabel == 'Rewards' else agent_game.losses
            x_rewards = range(1, len(data) + 1)
            y_rewards = data

            # plot results
            axes[i].set_title(f'{agent_name}: {xlabel} vs. {ylabel}')
            axes[i].set_xlabel(ylabel)
            axes[i].set_ylabel(xlabel)
            axes[i].plot(x_rewards, y_rewards)

        # plt.show()
        plt.savefig(f'{name}{xlabel}_{ylabel}_{env_name}_{level}.png', bbox_inches='tight')

    if train:
        # train step
        games = {}
        for agent in agents:
            # reduce episodes
            if level == 'super2' or level == 'super3':
                agent.config['episodes'] = 600

            print(f'==== {type(agent).__name__} ====')
            game = HighwayEnv(gym_env_name=env_name, model=agent, config_level=config, train_mode=True, load_model=load_model, debug=True, stochastic=stochastic)
            game.play()
            game.model.save_model()
            game.env.close()
            games[type(agent).__name__] = game
            print(f'==== END {type(agent).__name__} ====\n')
        if plot:
            plot_results(games, xlabel='Rewards', ylabel='Episodes', level=level)
            plot_results(games, xlabel='Losses', ylabel='Episodes', level=level)
    else:
        # evaluation step
        for agent in agents:
            print(f'==== {type(agent).__name__} - {env_name} ====')
            game = HighwayEnv(gym_env_name=env_name, model=agent, config_level=config, train_mode=False, load_model=True, debug=False, stochastic=stochastic)
            game.play()
            game.env.close()
            print(f'==== END {type(agent).__name__} ====\n')


# initial variables
render = False


if __name__ == "__main__":
    # run experiment 1
    # run_experiment(env_name='highway-fast-v0', agents=AgentFactory.get_easy_agents(), config=config1, train=True, plot=False, load_model=True, stochastic=True)
    run_experiment(env_name='highway-fast-v0', agents=AgentFactory.get_easy_agents(), config=config1, train=False, plot=False, load_model=True)

    # run experiment 2
    # run_experiment(env_name='highway-fast-v0', agents=AgentFactory.get_medium_agents(), config=config2, train=True, plot=False, load_model=True, level='medium')
    run_experiment(env_name='highway-fast-v0', agents=AgentFactory.get_medium_agents(), config=config2, train=False, plot=False, load_model=True)

    # run experiment 3
    # run_experiment(env_name='highway-fast-v0', agents=AgentFactory.get_super_agents(), config=config3, train=True, plot=False, load_model=True, level='super')
    run_experiment(env_name='highway-fast-v0', agents=AgentFactory.get_super_agents(), config=config3, train=False, plot=False, load_model=True)

    # run_experiment(env_name='merge-v0', agents=AgentFactory.get_super_agents(), config=config3, train=True, plot=False, load_model=True, level='super2')
    run_experiment(env_name='merge-v0', agents=AgentFactory.get_super_agents(), config=config3, train=False, plot=False, load_model=True)

    # run_experiment(env_name='roundabout-v0', agents=AgentFactory.get_super_agents(), config=config3, train=True, plot=False, load_model=True, level='super3')
    run_experiment(env_name='roundabout-v0', agents=AgentFactory.get_super_agents(), config=config3, train=False, plot=False, load_model=True)
