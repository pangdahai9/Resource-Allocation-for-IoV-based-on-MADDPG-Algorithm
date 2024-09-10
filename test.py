import torch
import torch.optim as optim
import numpy as np
from vehicular_env import VehicularEnv
from models import Actor, Critic
from replay_buffer import ReplayBuffer
from utils import soft_update, hard_update
from config import config
from cost_functions import calculate_operating_cost, total_system_cost, calculate_urlcc_qos_violation_cost, calculate_embb_qos_violation_cost, total_system_cost_all
import matplotlib.pyplot as plt
import pickle

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize environment
env = VehicularEnv(config['num_bs'])

def state_to_tensor(state, num_vehicles):
    vehicles = state['vehicles']
    bs = state['bs']

    max_x = 400
    max_y = 400

    vehicle_states = []
    for vehicle in vehicles:
        vehicle_states.extend([vehicle['x'] / max_x, vehicle['y'] / max_y, vehicle['c_s'] / 100, vehicle['c_c'] / 100, vehicle['d'] / 100, vehicle['z']])

    bs_states = []
    for b in bs:
        bs_states.extend([b['k'] / 100, b['l'] / 100])

    if len(vehicles) < num_vehicles:
        padding = [0] * (6 * (num_vehicles - len(vehicles)))
        state_list = vehicle_states + padding + bs_states
    else:
        state_list = vehicle_states + bs_states

    return torch.tensor(state_list, dtype=torch.float32).to(device)

def distance(vehicle, bs):
    return np.sqrt((vehicle['x'] - bs['x']) ** 2 + (vehicle['y'] - bs['y']) ** 2)

def initialize_distance_based_action(state, num_vehicles, num_bs):
    vehicles = state['vehicles']
    bs = state['bs']
    actions = []

    bs_load = [0] * num_bs
    bs_capacity = [7, 3, 3]

    for vehicle_index in range(num_vehicles):
        distances = [distance(vehicles[vehicle_index], bs[i]) for i in range(num_bs)]
        sorted_indices = np.argsort(distances)

        bs_actions = []
        for bs_index in sorted_indices:
            if bs_load[bs_index] < bs_capacity[bs_index]:
                q = 1
                bs_load[bs_index] += 1
                break
            else:
                q = 0

        b = np.random.uniform(0.1, 1) if q == 1 else 0
        f = np.random.uniform(0.1, 1) if q == 1 else 0

        for bs_index in range(num_bs):
            bs_actions.append({'q': q, 'b': b, 'f': f})

        actions.append(bs_actions)

    return actions

def add_gaussian_noise(action, noise_std):
    if isinstance(action, float):
        noise = np.random.normal(0, noise_std)
    else:
        noise = np.random.normal(0, noise_std, size=np.shape(action))
    return action + noise

def action_from_tensor(action_tensor, num_vehicles, num_bs, noise_std):
    action_list = action_tensor.view(-1).tolist()
    actions = []

    num_elements = num_vehicles * num_bs

    q_list = action_list[:num_elements]
    b_list = action_list[num_elements:2 * num_elements]
    f_list = action_list[2 * num_elements:]

    bs_load = [0] * num_bs
    bs_capacity = [7, 3, 3]

    for vehicle_index in range(num_vehicles):
        bs_actions = []
        q_values = []

        for bs_index in range(num_bs):
            q = add_gaussian_noise(q_list.pop(0), noise_std)
            b = max(0.1, add_gaussian_noise(b_list.pop(0), noise_std))
            f = max(0.1, add_gaussian_noise(f_list.pop(0), noise_std))
            q_values.append(q)
            bs_actions.append({'q': q, 'b': b, 'f': f})

        max_q_index = np.argmax(q_values)
        if bs_load[max_q_index] < bs_capacity[max_q_index]:
            for bs_index in range(num_bs):
                if bs_index != max_q_index:
                    bs_actions[bs_index]['q'] = 0
                    bs_actions[bs_index]['b'] = 0
                    bs_actions[bs_index]['f'] = 0
                else:
                    bs_actions[bs_index]['q'] = 1
                    bs_load[bs_index] += 1
        else:
            for bs_index in range(num_bs):
                bs_actions[bs_index]['q'] = 0
                bs_actions[bs_index]['b'] = 0
                bs_actions[bs_index]['f'] = 0

        actions.append(bs_actions)

    for bs_index in range(num_bs):
        total_b = sum([actions[vehicle_index][bs_index]['b'] for vehicle_index in range(num_vehicles)])
        total_f = sum([actions[vehicle_index][bs_index]['f'] for vehicle_index in range(num_vehicles)])

        if total_b > 1:
            scale_factor_b = 1 / total_b
            for vehicle_index in range(num_vehicles):
                actions[vehicle_index][bs_index]['b'] *= scale_factor_b

        if total_f > 1:
            scale_factor_f = 1 / total_f
            for vehicle_index in range(num_vehicles):
                actions[vehicle_index][bs_index]['f'] *= scale_factor_f

    return np.transpose(actions, (1, 0))

# Initialize Network
state_dim = env.state_dim
action_dim = env.action_dim

num_vehicles = config['num_vehicles']
num_bs = config['num_bs']

actor = Actor(state_dim, num_vehicles, num_bs).to(device)
critic = Critic(state_dim, action_dim).to(device)
target_actor = Actor(state_dim, num_vehicles, num_bs).to(device)
target_critic = Critic(state_dim, action_dim).to(device)
hard_update(target_actor, actor)
hard_update(target_critic, critic)

# Initialize optimizer
actor_optimizer = optim.Adam(actor.parameters(), lr=config['actor_lr'])
critic_optimizer = optim.Adam(critic.parameters(), lr=config['critic_lr'])

def clip_gradients(optimizer, max_norm):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, max_norm)

# Initialize experience playback pool
replay_buffer = ReplayBuffer(config['replay_buffer_capacity'], state_dim, action_dim)

# Initialize related variables
cumulative_system_costs = [0]
cumulative_operating_costs = [0]

# Configure the sampling frequency and slicing window index
sampling_interval = 2
slicing_window_index = 5
start_episode_for_slicing_window = 50

# Record the list of abnormal data
low_cost_data = []
high_cost_data = []

# Record data with QoS violation cost lower than 1800
qos_violation_data = []

# Record the distribution of b and f
b_values = []
f_values = []

# Training loop
selected_slicing_windows = []
all_costs = []
q_values_mbs = []
q_values_sbs1 = []
q_values_sbs2 = []
rewards_mbs = []
rewards_sbs1 = []
rewards_sbs2 = []

initial_noise_std = config['initial_noise_std']
final_noise_std = config['final_noise_std']
decay_episodes = config['decay_episodes']

def get_noise_std(episode):
    if episode >= decay_episodes:
        return final_noise_std
    else:
        decay_rate = (initial_noise_std - final_noise_std) / decay_episodes
        return initial_noise_std - decay_rate * episode

for episode in range(config['num_episodes']):
    state = env.reset()
    episode_rewards = []

    state_dim = env.state_dim
    action_dim = env.action_dim

    episode_costs = []
    episode_operating_costs = []
    episode_q_values_mbs = []
    episode_q_values_sbs1 = []
    episode_q_values_sbs2 = []
    episode_rewards_mbs = []
    episode_rewards_sbs1 = []
    episode_rewards_sbs2 = []

    prev_state = state
    noise_std = get_noise_std(episode)
    prev_action = action_from_tensor(torch.zeros(action_dim), env.num_vehicles, env.num_bs, noise_std)

    for step in range(config['num_steps_per_episode']):
        state_tensor = state_to_tensor(state, config['num_vehicles'])
        action_tensor = actor(state_tensor)
        action = action_from_tensor(action_tensor, config['num_vehicles'], env.num_bs, noise_std)

        for i in range(len(action)):
            action[i] = action[i][:len(state['vehicles'])]

        next_state, rewards, dones = env.step(action)
        next_state_tensor = state_to_tensor(next_state, config['num_vehicles'])

        scaled_rewards = [reward / config['reward_scaling_factor'] for reward in rewards]

        for bs_actions in action:
            for bs_action in bs_actions:
                if bs_action['q'] == 1:
                    b_values.append(bs_action['b'])
                    f_values.append(bs_action['f'])

        max_reward = 2000
        total_cost = total_system_cost_all(state, action, prev_state, prev_action)
        reward = max_reward - total_cost

        if total_cost < config['low_cost_threshold']:
            reward += config['extra_reward']
        elif total_cost > config['high_cost_threshold']:
            reward -= config['extra_penalty']

        reward /= config['reward_scaling_factor']
        total_reward = reward

        reward_tensors = torch.tensor([reward] * env.num_bs, dtype=torch.float32).view(1, -1).to(device)
        done_tensors = torch.tensor(dones, dtype=torch.float32).view(1, -1).to(device)

        replay_buffer.push(state_tensor, action_tensor, reward_tensors, next_state_tensor, done_tensors)

        episode_rewards.append(total_reward)

        if len(replay_buffer) > config['batch_size']:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices = replay_buffer.sample(
                config['batch_size'])

            state_batch = torch.stack([item.clone().detach().to(device) for item in state_batch]).view(config['batch_size'], -1)
            action_batch = torch.stack([item.clone().detach().to(device) for item in action_batch]).view(config['batch_size'], -1)
            reward_batch = torch.stack([item.clone().detach().to(device) for item in reward_batch]).view(config['batch_size'], env.num_bs)
            next_state_batch = torch.stack([item.clone().detach().to(device) for item in next_state_batch]).view(config['batch_size'], -1)
            done_batch = torch.stack([item.clone().detach().to(device) for item in done_batch]).view(config['batch_size'], env.num_bs)

            for m in range(env.num_bs):
                with torch.no_grad():
                    next_action_batch = target_actor(next_state_batch)
                    target_q = reward_batch[:, m] + (1 - done_batch[:, m]) * config['gamma'] * target_critic(next_state_batch, next_action_batch)[:, m]

                current_q = critic(state_batch, action_batch)[:, m]

                critic_loss = torch.nn.functional.mse_loss(current_q, target_q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                clip_gradients(critic_optimizer, 0.5)
                critic_optimizer.step()

                if m == 0:
                    episode_q_values_mbs.append(current_q.mean().item())
                elif m == 1:
                    episode_q_values_sbs1.append(current_q.mean().item())
                elif m == 2:
                    episode_q_values_sbs2.append(current_q.mean().item())

            priorities = torch.abs(current_q - target_q).detach().cpu().numpy() + 1e-6
            replay_buffer.update_priorities(indices, priorities)

            episode_rewards_mbs.append(total_reward)
            episode_rewards_sbs1.append(total_reward)
            episode_rewards_sbs2.append(total_reward)

            for m in range(env.num_bs):
                actor_loss = -critic(state_batch, actor(state_batch))[:, m].mean()
                regularization_loss = 1e-3 * (torch.sum(actor(state_batch) ** 2))
                total_actor_loss = actor_loss + regularization_loss
                actor_optimizer.zero_grad()
                total_actor_loss.backward()
                clip_gradients(actor_optimizer, 0.5)
                actor_optimizer.step()

            soft_update(target_actor, actor, config['tau'])
            soft_update(target_critic, critic, config['tau'])

        prev_state = state
        prev_action = action
        state = next_state

    for m in range(env.num_bs):
        cost = total_system_cost(state, action, prev_state, prev_action, m)
        episode_costs.append(cost)

    total_reward = sum(episode_rewards)
    print(f"Episode: {episode}, Total Reward: {total_reward}")

    if total_reward < 1000 or total_reward > 1500:
        episode_data = {'episode': episode, 'total_reward': total_reward, 'state': state, 'action': action}
        if total_reward < 1000:
            low_cost_data.append(episode_data)
        else:
            high_cost_data.append(episode_data)

    avg_cost = sum(episode_costs) / len(episode_costs) if episode_costs else 0
    all_costs.append(avg_cost)

    if episode_q_values_mbs:
        q_values_mbs.append(np.mean(episode_q_values_mbs))
    else:
        q_values_mbs.append(0)

    if episode_q_values_sbs1:
        q_values_sbs1.append(np.mean(episode_q_values_sbs1))
    else:
        q_values_sbs1.append(0)

    if episode_q_values_sbs2:
        q_values_sbs2.append(np.mean(episode_q_values_sbs2))
    else:
        q_values_sbs2.append(0)

    rewards_mbs.append(np.mean(episode_rewards_mbs))
    rewards_sbs1.append(np.mean(episode_rewards_sbs1))
    rewards_sbs2.append(np.mean(episode_rewards_sbs2))

    print(f"Episode {episode + 1}/{config['num_episodes']}, Average Cost: {avg_cost}")

    if episode >= start_episode_for_slicing_window and (episode - start_episode_for_slicing_window) % sampling_interval == 0:
        selected_slicing_windows.append((episode - start_episode_for_slicing_window) // sampling_interval + 1)
        cumulative_system_costs.append(cumulative_system_costs[-1] + avg_cost)
        cumulative_operating_costs.append(cumulative_operating_costs[-1] + avg_cost)

plt.figure(figsize=(18, 10))

plt.subplot(231)
plt.plot(all_costs)
plt.xlabel('Episodes')
plt.ylabel('Average Total Cost')
plt.title('Average Total System Cost over Each Episode')

plt.subplot(232)
plt.plot(q_values_mbs)
plt.xlabel('Episodes')
plt.ylabel('Average Q-value')
plt.title('MBS agent')

plt.subplot(233)
plt.plot(q_values_sbs1)
plt.xlabel('Episodes')
plt.ylabel('Average Q-value')
plt.title('SBS 1 agent')

plt.subplot(234)
plt.plot(q_values_sbs2)
plt.xlabel('Episodes')
plt.ylabel('Average Q-value')
plt.title('SBS 2 agent')

plt.subplot(235)
plt.plot(rewards_mbs)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('MBS agent Reward over Each Episode')

plt.subplot(236)
plt.plot(rewards_sbs1)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('SBS 1 agent Reward over Each Episode')

plt.figure(figsize=(6, 5))
plt.plot(rewards_sbs2)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('SBS 2 agent Reward over Each Episode')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.hist(b_values, bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.xlabel('b values')
plt.ylabel('Frequency')
plt.title('Distribution of b values')

plt.subplot(122)
plt.hist(f_values, bins=50, alpha=0.75, color='green', edgecolor='black')
plt.xlabel('f values')
plt.ylabel('Frequency')
plt.title('Distribution of f values')

plt.tight_layout()
plt.show()

with open('qos_violation_data.pkl', 'wb') as f:
    pickle.dump(qos_violation_data, f)
