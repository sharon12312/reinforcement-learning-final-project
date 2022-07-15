import os
import gym
import numpy as np
import highway_env
from matplotlib import pyplot as plt
import h5py
import time

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Lambda, concatenate
from tensorflow.keras.optimizers import RMSprop
from keras import backend as K

from multiprocessing import *
from collections import deque
from config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ICMBuilder:
    def build_icm_model(self, state_shape, action_shape, lmd=1.0, beta=0.01):
        s_t0 = Input(shape=state_shape, name="state0")
        s_t1 = Input(shape=state_shape, name="state1")
        a_t = Input(shape=action_shape, name="action")

        reshape = Reshape(target_shape=(1,) + state_shape)
        fmap = self.build_feature_map((1,) + state_shape)

        f_t0 = fmap(reshape(s_t0))
        f_t1 = fmap(reshape(s_t1))

        act_hat = self.inverse_model()(f_t0, f_t1)
        f_t1_hat = self.forward_model()(f_t0, a_t)

        rwd_int = Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1), output_shape=(1,),
                         name="reward_intrinsic")([f_t1, f_t1_hat])
        loss_inv = Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1), output_shape=(1,))(
            [a_t, act_hat])
        loss0 = Lambda(lambda x: beta * x[0] + (1.0 - beta) * x[1], output_shape=(1,))([rwd_int, loss_inv])

        rwd = Input(shape=(1,))
        loss = Lambda(lambda x: (-lmd * x[0] + x[1]), output_shape=(1,))([rwd, loss0])

        return Model([s_t0, s_t1, a_t, rwd], loss)

    @staticmethod
    def build_feature_map(input_shape):
        model = Sequential()
        model.add(Conv2D(32, 3, 3, padding='same', input_shape=input_shape, activation='relu'))
        model.add(Conv2D(32, 3, 3, padding='same', activation='relu'))
        model.add(Conv2D(32, 3, 3, padding='same', activation='relu'))
        model.add(Flatten(name="feature"))
        return model

    @staticmethod
    def inverse_model(output_dim=5):
        def func(ft0, ft1):
            h = concatenate([ft0, ft1])
            h = Dense(256, activation='relu')(h)
            h = Dense(output_dim, activation='sigmoid')(h)
            return h
        return func

    @staticmethod
    def forward_model(output_dim=160):
        def func(ft, at):
            h = concatenate([ft, at])
            h = Dense(256, activation='relu')(h)
            h = Dense(output_dim, activation='linear')(h)
            return h
        return func

    @staticmethod
    def get_reward_intrinsic(model, x):
        return K.function([model.get_layer("state0").input,
                           model.get_layer("state1").input,
                           model.get_layer("action").input],
                          [model.get_layer("reward_intrinsic").output])(x)[0]


class LearningAgent(object):
    def __init__(self, action_space, batch_size=32, swap_freq=200):
        _, _, self.train_net, advantage = construct_network(observation_shape, action_space.n)
        self.icm = icm_builder.build_icm_model(screen, (action_space.n,))

        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99), loss=['mse', 'categorical_crossentropy'])
        self.icm.compile(optimizer="rmsprop", loss='mse')

        self.pol_loss = deque(maxlen=25)
        self.val_loss = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.batch_size = batch_size
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, action_space.n))
        self.counter = 0

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames

        values, policy = self.train_net.predict([last_observations, self.unroll])

        self.targets.fill(0.)
        advantage = rewards - values.flatten()
        self.targets[self.unroll, :] = actions.astype(np.float32)

        loss = self.train_net.train_on_batch([last_observations, advantage], [rewards, self.targets])
        loss_icm = self.icm.train_on_batch([last_observations[:, -2, ...], last_observations[:, -1, ...], actions, rewards.reshape((-1, 1))], np.zeros((self.batch_size,)))

        self.store_results(loss, values, loss_icm)
        self.swap_counter -= frames

        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True

        return False

    def store_results(self, loss, values, loss_icm):
        self.pol_loss.append(loss[2])
        self.val_loss.append(loss[1])
        self.values.append(np.mean(values))

        if self.counter % 256 == 0:
            print(f'Policy-Loss: {loss[2]} (Avg: {np.mean(self.pol_loss)}), '
                  f'Value-Loss: {loss[1]} (Avg: {np.mean(self.val_loss)}), '
                  f'ICM-Loss: {loss_icm}')


class ActingAgent(object):
    def __init__(self, num_action, n_step=8, discount=0.99):
        self.value_net, self.policy_net, self.load_net, _ = construct_network(observation_shape, num_action)
        self.icm = icm_builder.build_icm_model(screen, (num_action,))

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='mse')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # initial loss-weights
        self.icm.compile(optimizer="rmsprop", loss=lambda y_true, y_pred: y_pred)

        self.num_action = num_action
        self.observations = np.zeros(observation_shape)
        self.last_observations = np.zeros_like(self.observations)

        self.n_step_data = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount

    def init_episode(self, observation):
        for _ in range(past_range):
            self.save_observation(observation)

    def reset(self):
        self.n_step_data.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)
        self.n_step_data.appendleft([self.last_observations, action, reward])

        if terminal or len(self.n_step_data) >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict(self.observations[None, ...])[0]
            for i in range(len(self.n_step_data)):
                r = self.n_step_data[i][2] + self.discount * r
                mem_queue.put((self.n_step_data[i][0], self.n_step_data[i][1], r), timeout=20)
            self.reset()

    def choose_action(self, observation=None, eps=0.1):
        if np.random.rand(1) < eps:
            action_arr = np.random.rand(self.num_action)
            action = np.argmax(action_arr)
        elif observation is None:
            action_arr = self.policy_net.predict(self.observations[None, ...])[0]
            action = np.argmax(action_arr)
        else:
            action_arr = self.policy_net.predict([observation[None, ...]])[0]
            action = np.argmax(action_arr)

        return self.get_stochastic_action(action), action_arr

    def get_stochastic_action(self, action):
        _actions = list(range(self.num_action))
        choice = np.random.choice(np.arange(2), p=[0.15, 0.85])

        if choice == 0:
            _actions.remove(action)
            action = np.random.choice(_actions)

        return int(action)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -input_depth, axis=0)
        self.observations[-input_depth:, ...] = transform_screen(observation)


class A3CAgentICM(object):
    def __init__(self, config_level):
        self.config_level = config_level

    def run(self):
        queue_size = A3CConfig().get_property('queue_size')
        processes = A3CConfig().get_property('processes')

        manager = Manager()
        weight_dict = manager.dict()
        mem_queue = manager.Queue(queue_size)
        pool = Pool(processes + 1, self.init_worker)

        try:
            for i in range(processes):
                pool.apply_async(self.generate_experience_proc, (mem_queue, weight_dict, i, self.config_level))

            pool.apply_async(self.learn_proc, (mem_queue, weight_dict, self.config_level))
            pool.close()
            pool.join()

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

        return dict(weight_dict), processes

    def exploit(self):
        game = A3CConfig().get_property('game')
        model_path = A3CConfig().get_property('model_path')

        env = gym.make(game)
        env.configure(self.config_level)
        # env = wrap_env(env)

        agent = ActingAgent(env.action_space.n)
        agent.load_net.load_weights(model_path)

        for i in range(1):
            done = False
            episode_reward = 0
            observation = env.reset()
            agent.init_episode(observation)

            print(f'Episode {i + 1}')

            while not done:
                action, _ = agent.choose_action(observation, eps=0.0)
                observation, reward, done, _ = env.step(action)
                env.render()
                episode_reward += reward

            print(f'Episode reward: {episode_reward}')

        env.close()
        # show_video()

    @staticmethod
    def learn_proc(mem_queue, weight_dict, config_level):
        import os
        pid = os.getpid()
        os.environ[
            'THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0.3,compiledir=th_comp_learn,optimizer=fast_compile'
        print(f'{pid}:: Learning process')

        save_freq = A3CConfig().get_property('save_freq')
        learning_rate = A3CConfig().get_property('learning_rate')
        batch_size = A3CConfig().get_property('batch_size')
        checkpoint = A3CConfig().get_property('checkpoint')
        steps = A3CConfig().get_property('steps')
        swap_freq = A3CConfig().get_property('swap_freq')
        game = A3CConfig().get_property('game')
        model_path = A3CConfig().get_property('model_path')
        icm_model_path = A3CConfig().get_property('icm_model_path')

        env = gym.make(game)
        env.configure(config_level)

        agent = LearningAgent(env.action_space, batch_size=batch_size, swap_freq=swap_freq)

        if checkpoint > 0:
            print(f'{pid}:: Loading weights from file...')
            agent.train_net.load_weights(model_path)

        print(f'{pid}:: Setting weights in dict...')
        weight_dict['update'] = 0
        weight_dict['weights'] = agent.train_net.get_weights()
        weight_dict['weights_icm'] = agent.icm.get_weights()

        last_obs = np.zeros((batch_size,) + observation_shape)
        actions = np.zeros((batch_size, env.action_space.n), dtype=np.int32)
        rewards = np.zeros(batch_size)

        idx = 0
        agent.counter = checkpoint
        save_counter = checkpoint % save_freq + save_freq

        for _ in range(episodes):
            last_obs[idx, ...], actions[idx, ...], rewards[idx] = mem_queue.get(timeout=20)
            idx = (idx + 1) % batch_size

            if idx == 0:
                lr = max(1.0e-8, (steps - agent.counter) / steps * learning_rate)
                updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)

                if updated:
                    weight_dict['weights'] = agent.train_net.get_weights()
                    weight_dict['weights_icm'] = agent.icm.get_weights()
                    weight_dict['update'] += 1

            save_counter -= 1
            if save_counter < 0:
                save_counter += save_freq
                agent.train_net.save_weights(model_path, overwrite=True)
                agent.icm.save_weights(icm_model_path, overwrite=True)

    @staticmethod
    def generate_experience_proc(mem_queue, weight_dict, no, config_level):
        import os
        pid = os.getpid()
        os.environ[
            'THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,compiledir=th_comp_act_' + str(
            no)
        print(f'{pid}:: Process started')

        frames = 0
        batch_size = A3CConfig().get_property('batch_size')
        n_step = A3CConfig().get_property('n_step')
        with_reward = A3CConfig().get_property('with_reward')
        checkpoint = A3CConfig().get_property('checkpoint')
        game = A3CConfig().get_property('game')
        model_path = A3CConfig().get_property('model_path')
        icm_model_path = A3CConfig().get_property('icm_model_path')

        env = gym.make(game)
        env.configure(config_level)
        # env = wrap_env(env)

        agent = ActingAgent(env.action_space.n, n_step=n_step)

        if checkpoint > 0:
            print(f'{pid}:: Loaded weights from file...')
            agent.load_net.load_weights(model_path)
            agent.icm.load_weights(icm_model_path)
        else:
            import time
            while 'weights' not in weight_dict:
                time.sleep(0.1)

            agent.load_net.set_weights(weight_dict['weights'])
            agent.icm.set_weights(weight_dict['weights_icm'])
            print(f'{pid}:: Loaded weights from dict...')

        episode_reward_history = []
        rewards_arr = []
        last_update = 0
        eta = 1.0

        for _ in range(episodes):
            done = False
            episode_reward = 0
            episode_duration = 0
            observation = env.reset()
            obs_last = observation.copy()
            agent.init_episode(observation)

            while not done:
                frames += 1
                episode_duration += 1

                action, action_arr = agent.choose_action(eps=max(1.0 / (frames / 100.0 + 1.0), 0.05))
                observation, reward, done, _ = env.step(action)

                r_in = icm_builder.get_reward_intrinsic(agent.icm,
                                                        [transform_screen(obs_last), transform_screen(observation),
                                                         action_arr.reshape(1, -1)])
                total_reward = reward + eta * r_in[0] if with_reward else eta * r_in[0]
                episode_reward += total_reward

                agent.sars_data(action, total_reward, observation, done, mem_queue)
                obs_last = observation.copy()

                if frames % batch_size == 0:
                    update = weight_dict.get('update', 0)
                    if update > last_update:
                        last_update = update
                        agent.load_net.set_weights(weight_dict['weights'])
                        agent.icm.set_weights(weight_dict['weights_icm'])

            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            rewards_arr.append(np.mean(episode_reward_history))
            weight_dict[f'{str(no)}_rewards'] = rewards_arr.copy()

    @staticmethod
    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def construct_network(input_shape, output_shape):
    state = Input(shape=input_shape)
    h = Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu')(state)
    h = Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu')(h)
    h = Flatten()(h)

    h = Dense(256, activation='relu')(h)
    value = Dense(1, activation='linear', name='value')(h)
    policy = Dense(output_shape, activation='softmax', name='policy')(h)

    value_network = Model(inputs=state, outputs=value)
    policy_network = Model(inputs=state, outputs=policy)

    advantage = Input(shape=(1,))
    _train_network = Model(inputs=[state, advantage], outputs=[value, policy])

    return value_network, policy_network, _train_network, advantage


def transform_screen(data):
    return (data[-1, :] / 255)[None, ...]


def plot_a3c_icm_rewards_durations_results(results, processes=1):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(30, 16))

    for i in range(4):
        for j in range(4):
          x_rewards = range(1, len(results[f'{str(i)}_rewards']) + 1)
          y_rewards = results[f'{str(i)}_rewards']
          y_rewards = (y_rewards - np.min(y_rewards)) / (np.max(y_rewards) - np.min(y_rewards))
          axes[i][j].set_title(f'Process {i + 1}: A3C & ICM: Rewards vs. Episodes')
          axes[i][j].set_xlabel('Episodes')
          axes[i][j].set_ylabel('Rewards')
          axes[i][j].plot(x_rewards, y_rewards)

    plt.show()


# initial variables
IMG_H = A3CConfig().get_property('IMG_H')
IMG_W = A3CConfig().get_property('IMG_W')
screen = (IMG_H, IMG_W)
input_depth, past_range = 1, 4
observation_shape = (input_depth * past_range,) + screen
episodes = A3CConfig().get_property('episodes')
icm_builder = ICMBuilder()

if __name__ == "__main__":
    # # train
    # a3c_icm = A3CAgentICM(config1)
    # results, processes = a3c_icm.run()
    # plot_a3c_icm_rewards_durations_results(results, processes)

    # evaluation
    a3c_icm = A3CAgentICM(config1)
    a3c_icm.exploit()
