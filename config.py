class AgentConfig:
    def __init__(self):
        self.config_dict = {
            'episodes': 10000,
            'batch_size': 32,
            'IMG_W': 128,
            'IMG_H': 128,
            'num_actions': 5,
            'iterations': 500,
            'discount_factor': 0.99,
            'learning_rate': 0.00025
        }

    def set_property(self, key, value):
        self.config_dict[key] = value

    def get_property(self, key):
        return self.config_dict[key]


class EasyConfig(AgentConfig):
    def get_dqn_config(self):
        self.config_dict['model_weights_path'] = 'weights/ex1_w/dqn_weights_easy.h5'
        return self.config_dict

    def get_ddqn_config(self):
        self.config_dict['model_weights_path'] = 'weights/ex1_w/ddqn_weights_easy.h5'
        return self.config_dict


class MediumConfig(AgentConfig):
    def get_dqn_config(self):
        self.config_dict['model_weights_path'] = 'weights/ex2_w/dqn_weights_medium.h5'
        return self.config_dict

    def get_ddqn_config(self):
        self.config_dict['model_weights_path'] = 'weights/ex2_w/ddqn_weights_medium.h5'
        return self.config_dict


class SuperConfig(AgentConfig):
    def get_dqn_config(self):
        self.config_dict['model_weights_path'] = 'weights/ex3_w/dqn_weights_super.h5'
        return self.config_dict

    def get_ddqn_config(self):
        self.config_dict['model_weights_path'] = 'weights/ex3_w/ddqn_weights_super.h5'
        return self.config_dict


class A3CConfig:
    def __init__(self):
        self.config_dict = {
            'game': 'highway-fast-v0',
            'IMG_H': 128,
            'IMG_W': 128,
            'episodes': 1000000,
            'save_freq': 100,
            'learning_rate': 0.00001,
            'batch_size': 32,
            'checkpoint': 0,
            'steps': 1000,
            'swap_freq': 60,
            'n_step': 5,
            'with_reward': True,
            'queue_size': 256,
            'processes': 16,
            'model_path': 'weights/ex1_w/a3c_icm_weights.h5',
            'icm_model_path': 'weights/ex1_w/icm_a3c_icm_weights.h5'
        }

    def set_property(self, key, value):
        self.config_dict[key] = value

    def get_property(self, key):
        return self.config_dict[key]


config1 = {
    'duration': 500,
    'observation':
        {
            'vehicles_count': 20,
            'vehicles_density': 3,
            'type': 'GrayscaleObservation',
            'observation_shape': (128, 128),
            'stack_size': 4,
            'weights': [0.2989, 0.587, 0.114],
            'scaling': 1.75
        },
    'policy_frequency': 2
}

config2 = {
    'duration': 500,
    'initial_vehicle_count': 30,
    'lanes_count': 6,
    'observation': {
        'observation_shape': (128, 128),
        'scaling': 1.75,
        'stack_size': 4,
        'type': 'GrayscaleObservation',
        'vehicles_count': 150,
        'vehicles_density': 75,
        'weights': [0.2989, 0.587, 0.114]
    },
    'other_vehicles_type': 'highway_env.vehicle.behavior.AggressiveVehicle',
    'policy_frequency': 2
}

config3 = {
    'duration': 500,
    'observation': {
        'observation_shape': (128, 128),
        'scaling': 1.75,
        'stack_size': 4,
        'type': 'GrayscaleObservation',
        'vehicles_count': 30,
        'vehicles_density': 15,
        'weights': [0.2989, 0.587, 0.114]
    },
    'policy_frequency': 2
}
