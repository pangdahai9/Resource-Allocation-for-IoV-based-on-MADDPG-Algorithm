import numpy as np
import logging
from config import config

def calculate_operating_cost(state, action, bs_index):
    operating_cost = 0
    bs_load = 0
    for i, v in enumerate(state['vehicles']):
        if v['out_of_range']:
            continue  # 跳过超出范围的车辆

        if i < len(action[bs_index]):
            q_i_m = action[bs_index][i]['q']
            b_i_m = action[bs_index][i]['b']
            f_i_m = action[bs_index][i]['f']
            W_m = config['W_m'] if bs_index == 0 else config['W_s']
            c_c_m = config['c_c_m'] if bs_index == 0 else config['c_c_s']
            omega_o_m_b = config['omega_o_m_b'] if bs_index == 0 else config['omega_o_s_b']
            omega_o_m_f = config['omega_o_m_f'] if bs_index == 0 else config['omega_o_s_f']
            cost = q_i_m * (omega_o_m_b * b_i_m * W_m + omega_o_m_f * f_i_m * c_c_m)
            operating_cost += cost
            if q_i_m == 1:
                bs_load += 1

    # 添加基站负载惩罚
    load_penalty = bs_load * 10  # 惩罚因子可以调整
    operating_cost += load_penalty
    print(f"Total Operating Cost for BS {bs_index}: {operating_cost}")
    return operating_cost

def calculate_reconfiguration_cost(state, prev_state, action, prev_action, bs_index):
    reconfiguration_cost = 0

    current_b_resource = 0
    current_f_resource = 0
    prev_b_resource = 0
    prev_f_resource = 0

    omega_r_m_b = config['omega_r_m_b'] if bs_index == 0 else config['omega_r_s_b']
    omega_r_m_f = config['omega_r_m_f'] if bs_index == 0 else config['omega_r_s_f']
    W_m = config['W_m'] if bs_index == 0 else config['W_s']
    c_c_m = config['c_c_m'] if bs_index == 0 else config['c_c_s']

    for i, v in enumerate(state['vehicles']):
        if v['out_of_range']:
            continue  # 跳过超出范围的车辆

    for i, v in enumerate(state['vehicles']):
        if i < len(action[bs_index]):
            q_i_m = action[bs_index][i]['q']
            b_i_m = action[bs_index][i]['b']
            f_i_m = action[bs_index][i]['f']
            prev_q_i_m = prev_action[bs_index][i]['q']
            prev_b_i_m = prev_action[bs_index][i]['b']
            prev_f_i_m = prev_action[bs_index][i]['f']

            current_b_resource += q_i_m * b_i_m * W_m
            current_f_resource += q_i_m * f_i_m * c_c_m

            prev_b_resource += prev_q_i_m * prev_b_i_m * W_m
            prev_f_resource += prev_q_i_m * prev_f_i_m * c_c_m

    delta_b_resource = max(0, current_b_resource - prev_b_resource)
    delta_f_resource = max(0, current_f_resource - prev_f_resource)

    cost_b = omega_r_m_b * delta_b_resource
    cost_f = omega_r_m_f * delta_f_resource

    reconfiguration_cost += cost_b + cost_f
    print(
        f"BS {bs_index}, Current b_resource: {current_b_resource}, Prev b_resource: {prev_b_resource}, Delta_b: {delta_b_resource}, Cost_b: {cost_b}")
    print(
        f"BS {bs_index}, Current f_resource: {current_f_resource}, Prev f_resource: {prev_f_resource}, Delta_f: {delta_f_resource}, Cost_f: {cost_f}")

    print(f"Total Reconfiguration Cost for BS {bs_index}: {reconfiguration_cost}")
    return reconfiguration_cost


def calculate_urlcc_qos_violation_cost(state, actions, bs_index):
    qos_violation_cost = 0

    for i, v in enumerate(state['vehicles']):
        if v['out_of_range'] or v['z'] != 1:
            continue  # 跳过超出范围或不在URLLC切片的车辆

        c_s_i = v['c_s']
        c_c_i = v['c_c']
        d_i = v['d']
        b_i_m = None
        f_i_m = None

        # 确保车辆只分配到一个基站
        assigned_bs = None
        for m in range(len(state['bs'])):
            if i < len(actions[m]) and actions[m][i]['q'] == 1:
                assigned_bs = m
                b_i_m = actions[m][i]['b']
                f_i_m = actions[m][i]['f']
                break

        if assigned_bs is None:
            continue  # 如果车辆没有被分配到任何基站，跳过

        # 根据分配的基站类型设置W_m和c_c_m
        if assigned_bs == 0:  # MBS
            W_m = config['W_m']
            c_c_m = config['c_c_m']
        else:  # SBS
            W_m = config['W_s']
            c_c_m = config['c_c_s']

        h_i_m = v.get('h_i_m', 1e-6)  # 确保 h_i_m 存在
        R_i_m = b_i_m * W_m * h_i_m  # 在这里计算 R_i_m
        f_i_m_allocated = f_i_m * c_c_m

        # 防止除以零的错误
        R_i_m = R_i_m if R_i_m != 0 else 1e-6
        f_i_m_allocated = f_i_m_allocated if f_i_m_allocated != 0 else 1e-6

        T_i = (c_s_i / R_i_m) + (c_c_i / f_i_m_allocated)
        if T_i > d_i:
            violation_cost = config['omega_q_u']
        else:
            violation_cost = 0

        if assigned_bs == bs_index:
            qos_violation_cost += violation_cost

        print(f"Vehicle {i}: assigned to BS {assigned_bs}, z=1, c_s={c_s_i}, c_c={c_c_i}, d={d_i}, f_i_m={f_i_m}, "
              f"R_i_m={R_i_m}, b_i_m={b_i_m}, h_i_m={h_i_m}, T_i={T_i}, "
              f"f_i_m_allocated={f_i_m_allocated}, violation_cost={violation_cost}")

    print(f"Total URLCC Qos violation Cost for BS {bs_index}: {qos_violation_cost}")
    return qos_violation_cost

def calculate_embb_qos_violation_cost(state, actions, bs_index):
    qos_violation_cost = 0
    R_e = config['R_e']
    c_e = config['c_e']

    for i, v in enumerate(state['vehicles']):
        if v['out_of_range'] or v['z'] != 0:
            continue  # Skip vehicles that are out of range or not in eMBB slice

        c_s_i = v['c_s']
        c_c_i = v['c_c']
        d_i = v['d']
        b_i_m = None
        f_i_m = None

        # Ensure that the vehicle is assigned to only one base station
        assigned_bs = None
        for m in range(len(state['bs'])):
            if i < len(actions[m]) and actions[m][i]['q'] == 1:
                assigned_bs = m
                b_i_m = actions[m][i]['b']
                f_i_m = actions[m][i]['f']
                break

        if assigned_bs is None:
            continue  # If the vehicle is not assigned to any base station, skip

        # Set W_m and c_c_m according to the assigned BS type
        if assigned_bs == 0:  # MBS
            W_m = config['W_m']
            c_c_m = config['c_c_m']
        else:  # SBS
            W_m = config['W_s']
            c_c_m = config['c_c_s']

        h_i_m = v.get('h_i_m', 1e-6)
        R_i_m = b_i_m * W_m * h_i_m
        f_i_m_allocated = f_i_m * c_c_m

        # 防止除以零的错误
        R_i_m = R_i_m if R_i_m != 0 else 1e-6
        f_i_m_allocated = f_i_m_allocated if f_i_m_allocated != 0 else 1e-6

        R_i_m_allocated = actions[assigned_bs][i]['q'] * R_i_m
        if R_i_m_allocated < R_e or f_i_m_allocated < c_e:
            violation_cost = config['omega_q_e']
        else:
            violation_cost = 0

        if assigned_bs == bs_index:
            qos_violation_cost += violation_cost

        print(f"Vehicle {i}: assigned to BS {assigned_bs}, z=0, c_s={c_s_i}, c_c={c_c_i}, d={d_i}, f_i_m={f_i_m}, "
              f"R_i_m={R_i_m}, b_i_m={b_i_m}, h_i_m={h_i_m}, R_i_m_allocated={R_i_m_allocated}, "
              f"f_i_m_allocated={f_i_m_allocated}, violation_cost={violation_cost}")

    print(f"Total eMBB Qos violation Cost for BS {bs_index}: {qos_violation_cost}")
    return qos_violation_cost

def total_system_cost(state, action, prev_state, prev_action, bs_index):
    operating_cost = calculate_operating_cost(state, action, bs_index)
    reconfiguration_cost = calculate_reconfiguration_cost(state, prev_state, action, prev_action, bs_index)
    urlcc_cost = calculate_urlcc_qos_violation_cost(state, action, bs_index)
    embb_cost = calculate_embb_qos_violation_cost(state, action, bs_index)
    total_cost = operating_cost + reconfiguration_cost + urlcc_cost + embb_cost
    print(f"Total System Cost for BS {bs_index}: {total_cost}")
    return total_cost

def total_system_cost_all(state, action, prev_state, prev_action):
    total_cost = 0
    for m in range(len(state['bs'])):
        total_cost += total_system_cost(state, action, prev_state, prev_action, m)
    return total_cost