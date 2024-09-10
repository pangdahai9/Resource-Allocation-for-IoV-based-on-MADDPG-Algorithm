config = {
    'num_vehicles': 10,  # Maximum number of vehicles
    'num_bs': 3,  # 1 MBS + 2 SBS
    'actor_lr': 0.0001,  # Actor learning rate
    'critic_lr': 0.001,  # Critic learning rate
    'gamma': 0.9,
    'tau': 0.01,
    'batch_size': 32,
    'replay_buffer_capacity': 20000,
    'num_episodes': 4000,
    'num_steps_per_episode': 50,  # Maximum steps per episode
    'num_agents': 3,
    'omega_o_m_b': 15 / 10,
    'omega_o_s_b': 20 / 2,
    'omega_o_m_f': 15 / 10000,
    'omega_o_s_f': 20 / 2000,
    'omega_r_m_b': 50 / 10,
    'omega_r_s_b': 60 / 2,
    'omega_r_m_f': 50 / 10000,
    'omega_r_s_f': 60 / 2000,
    'R_e': 2,  # Minimum data rate requirement for eMBB slices in Mbit/s
    'c_e': 0.2,  # Minimum computational resource requirement for eMBB slices
    'omega_q_u': 200,
    'omega_q_e': 200,
    'W_m': 10,  # Spectrum resource unit of MBS in MHz
    'W_s': 2,  # Spectrum resource unit of SBS in MHz
    'c_c_m': 10*1000,  # Computational resource unit of MBS in MHz
    'c_c_s': 2*1000,  # Computational resource unit of SBS in MHz
    'low_cost_threshold': 200,  # Threshold for low system cost
    'high_cost_threshold': 1500,  # Threshold for high system cost
    'extra_reward': 100,  # Extra reward
    'extra_penalty': -100,  # Extra penalty
    'reward_scaling_factor': 1,
    'initial_noise_std': 1.0,  # Initial standard deviation of Gaussian noise
    'final_noise_std': 0.0001,  # Final standard deviation of Gaussian noise
    'decay_episodes': 1500  # Number of episodes to decay noise standard deviation to its final value
}
