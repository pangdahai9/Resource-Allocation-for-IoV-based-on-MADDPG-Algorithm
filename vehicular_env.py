import numpy as np
from cost_functions import calculate_operating_cost, calculate_reconfiguration_cost, calculate_urlcc_qos_violation_cost, calculate_embb_qos_violation_cost
from config import config

def calculate_channel_gain(distance, bs_type):
    min_distance = 1e-3
    distance = max(distance, min_distance)

    if bs_type == 'MBS':
        channel_gain = 10 ** ((-30 - 35 * np.log10(distance)) / 10 )
    elif bs_type == 'SBS':
        channel_gain = 10 ** ((-40 - 35 * np.log10(distance)) / 10 )
    else:
        raise ValueError("Unknown base station type")

    # 打印信道增益
    print(f"distance={distance}, bs_type={bs_type}, channel_gain1={channel_gain}")
    return channel_gain

class VehicularEnv:
    def __init__(self, num_bs):
        self.num_vehicles = config['num_vehicles']
        self.num_bs = num_bs
        self.MBS_bandwidth = config['W_m']  # MHz
        self.SBS_bandwidth = config['W_s']  # MHz
        self.gamma = config['gamma']  # discount factor
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()
        self.reset()

    def reset(self):
        self.state = self._initialize_state()
        self.prev_state = self.state  # Initialize prev_state
        self.prev_action = self._initialize_action()  # Initialize prev_action
        return self.state

    def _initialize_state(self):
        def is_within_circle(x, y, center_x, center_y, radius):
            return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

        state = {
            'vehicles': [],
            'bs': [
                      {'x': 200, 'y': 200, 'k': 0, 'l': 0, 'type': 'MBS'}
                  ] + [
                      {'x': 100, 'y': 200, 'k': 0, 'l': 0, 'type': 'SBS'},
                      {'x': 300, 'y': 200, 'k': 0, 'l': 0, 'type': 'SBS'}
                  ]
        }

        while len(state['vehicles']) < self.num_vehicles:
            x = np.random.uniform(0, 400)
            y = np.random.uniform(0, 400)
            if is_within_circle(x, y, 200, 200, 200):
                vehicle = {
                    'x': x,
                    'y': y,
                    'c_s': np.random.uniform(0.5/1000, 1/1000),  # Mbits
                    'c_c': np.random.uniform(50, 100),  # MHz
                    'd': np.random.uniform(10/100, 50/100),  # s
                    'z': np.random.choice([0, 1], p=[0.5, 0.5]),
                    'out_of_range': False
                }

                state['vehicles'].append(vehicle)

        return state

    def _initialize_action(self):
        return [
            [{'q': 0, 'b': 0, 'f': 0} for _ in range(self.num_vehicles)]
            for _ in range(self.num_bs)
        ]

    def step(self, actions):
        self._update_vehicle_positions()
        next_state = self._get_next_state(actions)
        rewards = []
        dones = [False] * self.num_bs

        for vehicle_index in range(self.num_vehicles):
            if self.state['vehicles'][vehicle_index]['out_of_range']:
                continue  # Skip out of range vehicles

            q_i_0 = np.random.rand()

            # Ensure that there are enough elements in the actions list
            q_i_1 = actions[1][vehicle_index]['q'] if len(actions) > 1 and vehicle_index < len(actions[1]) else 0
            q_i_2 = actions[2][vehicle_index]['q'] if len(actions) > 2 and vehicle_index < len(actions[2]) else 0

            if q_i_0 >= q_i_1 and q_i_0 >= q_i_2:
                actions[0][vehicle_index]['unload_to'] = 'MBS'
                actions[0][vehicle_index]['q'] = 1  # Set q_ (i, 0) ^ t to 1
                if len(actions) > 1 and vehicle_index < len(actions[1]):
                    actions[1][vehicle_index]['q'] = 0  # Set q_ (i, 1) ^ t to 0
                if len(actions) > 2 and vehicle_index < len(actions[2]):
                    actions[2][vehicle_index]['q'] = 0  # Set q_ (i, 2) ^ t to 0
            elif q_i_1 >= q_i_0 and q_i_1 >= q_i_2:
                actions[1][vehicle_index]['unload_to'] = 'SBS1'
                actions[1][vehicle_index]['q'] = 1
                if len(actions) > 0 and vehicle_index < len(actions[0]):
                    actions[0][vehicle_index]['q'] = 0
                if len(actions) > 2 and vehicle_index < len(actions[2]):
                    actions[2][vehicle_index]['q'] = 0
            else:
                actions[2][vehicle_index]['unload_to'] = 'SBS2'
                actions[2][vehicle_index]['q'] = 1
                if len(actions) > 0 and vehicle_index < len(actions[0]):
                    actions[0][vehicle_index]['q'] = 0
                if len(actions) > 1 and vehicle_index < len(actions[1]):
                    actions[1][vehicle_index]['q'] = 0

        for m in range(self.num_bs):
            U_t_m = self.calculate_cost_for_bs(m, actions)
            rewards.append(-U_t_m)
            if self._check_done():
                dones[m] = True

        self.prev_state = self.state  # 更新 prev_state
        self.prev_action = actions  # 更新 prev_action
        self.state = next_state  # 更新当前状态
        return next_state, rewards, dones

    def _update_vehicle_positions(self):
        for vehicle in self.state['vehicles']:
            if vehicle['out_of_range']:
                vehicle['x'] = 0
                vehicle['y'] = 0
                vehicle['c_s'] = 0
                vehicle['c_c'] = 0
                vehicle['d'] = 0
                vehicle['z'] = 0
                continue

            direction = np.random.uniform(0, 2 * np.pi)
            speed = 10
            vehicle['x'] += speed * np.cos(direction)
            vehicle['y'] += speed * np.sin(direction)
            if np.sqrt((vehicle['x'] - 200) ** 2 + (vehicle['y'] - 200) ** 2) > 200:
                vehicle['out_of_range'] = True  # 标记车辆超出范围
                vehicle['x'] = 0
                vehicle['y'] = 0
                vehicle['c_s'] = 0
                vehicle['c_c'] = 0
                vehicle['d'] = 0
                vehicle['z'] = 0
            else:
                vehicle['x'] = np.clip(vehicle['x'], 0, 400)
                vehicle['y'] = np.clip(vehicle['y'], 0, 400)

    def _get_next_state(self, actions):
        next_state = self.state.copy()
        for bs_index, bs_actions in enumerate(actions):
            next_state['bs'][bs_index]['k'] = 0
            next_state['bs'][bs_index]['l'] = 0
            for vehicle_index, action in enumerate(bs_actions):
                if vehicle_index >= len(next_state['vehicles']):
                    continue
                if next_state['vehicles'][vehicle_index]['out_of_range']:
                    next_state['vehicles'][vehicle_index]['x'] = 0
                    next_state['vehicles'][vehicle_index]['y'] = 0
                    next_state['vehicles'][vehicle_index]['c_s'] = 0
                    next_state['vehicles'][vehicle_index]['c_c'] = 0
                    next_state['vehicles'][vehicle_index]['d'] = 0
                    next_state['vehicles'][vehicle_index]['z'] = 0
                    continue  # Skip out of range vehicles

                distance = np.sqrt(
                    (next_state['vehicles'][vehicle_index]['x'] - next_state['bs'][bs_index]['x']) ** 2 +
                    (next_state['vehicles'][vehicle_index]['y'] - next_state['bs'][bs_index]['y']) ** 2
                )
                channel_gain = calculate_channel_gain(distance, next_state['bs'][bs_index]['type'])
                next_state['vehicles'][vehicle_index]['channel_gain'] = channel_gain

                if next_state['bs'][bs_index]['type'] == 'MBS':
                    h_i_m = self.calculate_spectrum_efficiency(channel_gain, 'MBS')
                else:
                    interfered_channel_gains = [
                        calculate_channel_gain(
                            np.sqrt(
                                (next_state['vehicles'][vehicle_index]['x'] - next_state['bs'][other_bs_index][
                                    'x']) ** 2 +
                                (next_state['vehicles'][vehicle_index]['y'] - next_state['bs'][other_bs_index][
                                    'y']) ** 2
                            ), 'SBS'
                        )
                        for other_bs_index in range(len(next_state['bs'])) if other_bs_index != bs_index
                    ]
                    h_i_m = self.calculate_spectrum_efficiency(channel_gain, 'SBS', interfered_channel_gains)

                next_state['vehicles'][vehicle_index]['h_i_m'] = h_i_m

                if next_state['bs'][bs_index]['type'] == 'MBS':
                    next_state['bs'][bs_index]['k'] += action['b'] * self.MBS_bandwidth
                    next_state['bs'][bs_index]['l'] += action['f'] * 100
                else:
                    next_state['bs'][bs_index]['k'] += action['b'] * self.SBS_bandwidth
                    next_state['bs'][bs_index]['l'] += action['f'] * 100
                next_state['vehicles'][vehicle_index]['channel_gain'] = channel_gain
            return next_state

    def calculate_spectrum_efficiency(self, channel_gain, bs_type, interfered_channel_gains=None):
        P = 10 ** (23 / 10) * 1e-3  # Set the transmit power to 23dBm
        noise_power_density = 10 ** (-114 / 10) * 1e-3  # -114dBm noise power spectral density

        # Ensure channel_gain is non-negative
        # channel_gain = max(channel_gain, 1e-6)

        if bs_type == 'MBS':
            h_i_m = np.log2(1 + (P * channel_gain) / noise_power_density)
        elif bs_type == 'SBS':
            interference = sum([P * gain for gain in interfered_channel_gains]) if interfered_channel_gains else 0
            h_i_m = np.log2(1 + (P * channel_gain) / (interference + noise_power_density))
        else:
            raise ValueError("Unknown base station type")

        print(f"channel_gain2={channel_gain}, bs_type={bs_type}, h_i_m={h_i_m}")

        return h_i_m

    def calculate_cost_for_bs(self, bs_index, actions):
        operating_cost = calculate_operating_cost(self.state, actions, bs_index)
        reconfiguration_cost = calculate_reconfiguration_cost(self.state, self.prev_state, actions, self.prev_action,
                                                              bs_index)
        urlcc_cost = calculate_urlcc_qos_violation_cost(self.state, actions, bs_index)
        embb_cost = calculate_embb_qos_violation_cost(self.state, actions, bs_index)
        total_cost = operating_cost + reconfiguration_cost + urlcc_cost + embb_cost
        return total_cost

    def _check_done(self):
        return False

    def _calculate_state_dim(self):
        vehicle_state_dim = 6  # Dimension of vehicle status
        bs_state_dim = 2  # Dimension of BS status
        return self.num_vehicles * vehicle_state_dim + self.num_bs * bs_state_dim

    def _calculate_action_dim(self):
        vehicle_action_dim = 3  # Dimension of vehicle action
        return self.num_vehicles * vehicle_action_dim * self.num_bs

    def get_observation(self, bs_index):
        observation = {
            'vehicles': [
                {
                    'x': v['x'],
                    'y': v['y'],
                    'c_s': v['c_s'],
                    'c_c': v['c_c'],
                    'd': v['d'],
                    'z': v['z']
                } for v in self.state['vehicles']
            ],
            'k': self.state['bs'][bs_index]['k'],
            'l': self.state['bs'][bs_index]['l']
        }
        return observation