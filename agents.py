import numpy as np
import tensorflow as tf

from config import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import Huber


class DQN:

    def __init__(self, config=None):
        self.config = config
        self.epsilon, self.min_epsilon = 1.0, 0.1
        self.epsilon_interval = (self.epsilon - self.min_epsilon)

        self.train_network = self.build_network()
        self.predict_network = self.build_network()
        self.count = 0

        self.optimizer = RMSprop(learning_rate=self.config['learning_rate'], rho=0.95)
        self.loss_function = Huber()

        # self.optimizer = Adam(learning_rate=self.config['learning_rate'], clipnorm=1.0)
        # self.loss_history = []

    def train(self, data):
        self.count += 1
        current_states, actions, next_states, rewards, done = data
        current_q_values = self.predict(current_states)
        next_q_values = self.get_prediction(next_states)

        current_q_values = self.update_q_value(rewards, current_q_values, next_q_values, actions, done)
        current_states = np.reshape(current_states, newshape=(self.config['batch_size'], self.config['IMG_H'], self.config['IMG_W'], 4)) / 255

        # history = self.train_network.fit(current_states, current_q_values)
        # self.loss_history.append(np.mean(history.history['loss']))

        loss = self.train_step(current_states, current_q_values)
        # self.loss_history.append(np.mean(loss))
        print(f'Loss: {np.mean(loss)}')

        return np.mean(loss)

    def update_q_value(self, rewards, current_q_values, next_q_values, actions, done):
        current_q_values = current_q_values.numpy()
        next_max_q_values = np.max(next_q_values, axis=1)
        new_q_values = rewards + self.config['discount_factor'] * next_max_q_values

        for i in range(len(current_q_values)):
            current_q_values[i, actions[i]] = new_q_values[i] if not done[i] else rewards[i]

        return current_q_values

    def predict(self, states):
        states = np.reshape(states, newshape=(states.shape[0], self.config['IMG_H'], self.config['IMG_W'], 4)) / 255
        prediction = self.train_network(states)
        return prediction

    def get_prediction(self, states):
        states = np.reshape(states, newshape=(states.shape[0], self.config['IMG_H'], self.config['IMG_W'], 4)) / 255
        prediction = self.predict_network(states)
        return prediction

    def get_action(self, state, stochastic=True, exploit=False):
        if np.random.random() > self.epsilon or exploit:
            _action = self.predict(np.expand_dims(state, axis=0))
            action = np.argmax(_action)
        else:
            action = np.random.randint(0, self.config['num_actions'])

        return self.get_stochastic_action(action) if stochastic else action

    def get_stochastic_action(self, action):
        _actions = list(range(self.config['num_actions']))
        choice = np.random.choice(np.arange(2), p=[0.15, 0.85])

        if choice == 0:
            _actions.remove(action)
            action = np.random.choice(_actions)

        return int(action)

    def update_prediction(self):
        self.predict_network.set_weights(self.train_network.get_weights())
        self.save_model()

    def _build_network(self):
        model = Sequential()

        # Conv Layers
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', activation='relu', input_shape=(self.config['IMG_H'], self.config['IMG_W'], 4)))
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.config['num_actions'], activation='linear'))

        # model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.config['learning_rate'], rho=0.95))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.config['learning_rate'], clipnorm=1.0))

        return model

    def build_network(self):
        inputs = tf.keras.layers.Input(shape=(self.config['IMG_H'], self.config['IMG_W'], 4))

        # Convolutions on the frames on the screen
        layer1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = tf.keras.layers.Flatten()(layer3)
        layer5 = tf.keras.layers.Dense(512, activation="relu")(layer4)
        action = tf.keras.layers.Dense(self.config['num_actions'], activation="linear")(layer5)

        return tf.keras.Model(inputs=inputs, outputs=action)

    def save_model(self):
        self.train_network.save_weights(self.config['model_weights_path'])
        print('The model has been successfully saved ...')

    def load_model(self):
        self.train_network.load_weights(self.config['model_weights_path'])
        self.predict_network.load_weights(self.config['model_weights_path'])
        print('The model has been successfully loaded ...')

    def summary(self):
        self.train_network.summary()

    @staticmethod
    def loss(ground_truth, prediction):
        loss = tf.keras.losses.mean_squared_error(ground_truth, prediction)
        return loss

    @tf.function
    def train_step(self, states, actions):
        with tf.GradientTape() as tape:
            predictions = self.train_network(states)
            loss = self.loss_function(actions, predictions)
            # loss = self.loss(actions, predictions)

        gradients = tape.gradient(loss, self.train_network.trainable_variables)
        gradients = [tf.clip_by_norm(gradient, 10) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.train_network.trainable_variables))
        return loss


class DDQN(DQN):

    def __init__(self, config):
        super().__init__(config)

    def train(self, data):
        self.count += 1
        current_states, _actions, next_states, rewards, done = data
        current_q_values = self.predict(current_states)
        next_q_values = self.predict(next_states)
        next_target_q_values = self.get_prediction(next_states)

        current_q_values = self.update_q_values(rewards, current_q_values, next_q_values, next_target_q_values, _actions, done)
        current_states = np.reshape(current_states, newshape=(self.config['batch_size'], self.config['IMG_H'], self.config['IMG_W'], 4)) / 255

        # history = self.train_network.fit(current_states, current_q_values)
        # self.loss_history.append(np.mean(history.history['loss']))

        loss = self.train_step(current_states, current_q_values)
        # self.loss_history.append(np.mean(loss))
        print(f'Loss: {np.mean(loss)}')

        return np.mean(loss)

    def update_q_values(self, rewards, current_q_values, next_q_values, next_target_q_values, _actions, done):
        current_q_values = current_q_values.numpy()
        next_target_q_values = next_target_q_values.numpy()
        indices = np.arange(next_target_q_values.shape[0])
        max_actions = np.argmax(next_q_values, axis=1)
        new_q_values = rewards + self.config['discount_factor'] * next_target_q_values[indices, max_actions]

        for i in range(len(current_q_values)):
            if not done[i]:
                current_q_values[i, _actions[i]] = new_q_values[i]
            else:
                current_q_values[i, _actions[i]] = rewards[i]

        return current_q_values


class AgentFactory:
    @classmethod
    def get_easy_agents(cls):
        dqn = DQN(EasyConfig().get_dqn_config())
        ddqn = DDQN(EasyConfig().get_ddqn_config())
        return [dqn, ddqn]

    @classmethod
    def get_medium_agents(cls):
        dqn = DQN(MediumConfig().get_dqn_config())
        ddqn = DDQN(MediumConfig().get_ddqn_config())
        return [dqn, ddqn]

    @classmethod
    def get_super_agents(cls):
        dqn = DQN(SuperConfig().get_dqn_config())
        ddqn = DDQN(SuperConfig().get_ddqn_config())
        return [dqn, ddqn]
