import gymnasium as gym
import numpy as np


class Environment(object):
    def __init__(self, map=None, is_slippery=False, *args, **kwargs):
        # Initialise the environment
        self.environment = gym.make(
            'FrozenLake-v1',
            desc=map,
            is_slippery=is_slippery,
            render_mode="human",
        )

    def close(self):
        self.environment.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reset(self, *args, **kwargs):
        observation, info = self.environment.reset(*args, **kwargs)
        return observation, info

    def sample_action_space(self):
        return self.environment.action_space.sample()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.environment.step(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        self.environment.render()

    def test(self):
        episode_over = False
        total_reward = 0

        while not episode_over:
            action = self.sample_action_space()

            observation, reward, terminated, truncated, info = self.step(action)
            print("================== staging ==================")
            print(f"observation: {observation}")
            print(f"reward: {reward}")
            print(f"terminated: {terminated}")
            print(f"truncated: {truncated}")
            print(f"info: {info}")

            total_reward += reward
            episode_over = terminated or truncated


def build_transfer_function(map):
    rows = len(map)
    cols = len(map[0])
    transfer_function = np.ones((rows, cols, 4))

    for _row in range(rows):
        # to left boundary
        transfer_function[_row, 0, 0] = 0
        # to right boundary
        transfer_function[_row, -1, 2] = 0

    for _col in range(cols):
        # to upper boundary
        transfer_function[0, _col, 3] = 0
        # to bottom boundary
        transfer_function[-1, _col, 1] = 0

    return transfer_function


def predict(map, pi, action_value, gamma=0.9, *args, **kwargs):
    new_action_value = action_value.copy()
    transfer_function = build_transfer_function(map)

    for _row in range(len(map)):
        for _col in range(len(map[0])):
            if map[_row][_col] in ('H', 'G'):
                continue

            # q(s) = ∑{s', r} p(s', r | s, a) [r + γ v(s')]
            for _action in (0, 1, 2, 3):
                # get next state
                if _action == 0:
                    next_row, next_col = _row, _col - 1
                elif _action == 1:
                    next_row, next_col = _row + 1, _col
                elif _action == 2:
                    next_row, next_col = _row, _col + 1
                else:
                    next_row, next_col = _row - 1, _col

                if next_row >= 8 or next_row < 0 or next_col >= 8 or next_col < 0:
                    continue

                if (next_row, next_col) == (7, 7):
                    # if reach the goal
                    reward = 1.0
                else:
                    # not the goal
                    reward = 0.0

                next_value = np.max(action_value[next_row, next_col])
                new_action_value[_row, _col, _action] = transfer_function[_row, _col, _action] * (reward + gamma * next_value)

    return new_action_value


def improve(map, pi, action_value, gamma=0.9, *args, **kwargs):
    new_pi = pi.copy()
    transfer_function = build_transfer_function(map)

    for _row in range(len(map)):
        for _col in range(len(map[0])):
            if map[_row][_col] in ('H', 'G'):
                continue

            # q*(s, a) = argmax_a ∑{s', r} p(s', r | s, a) [r + γ v(s')]
            action_transform_score = np.zeros([4])

            for _action in (0, 1, 2, 3):
                # get next state
                if _action == 0:
                    next_row, next_col = _row, _col - 1
                elif _action == 1:
                    next_row, next_col = _row + 1, _col
                elif _action == 2:
                    next_row, next_col = _row, _col + 1
                else:
                    next_row, next_col = _row - 1, _col

                if next_row >= 8 or next_row < 0 or next_col >= 8 or next_col < 0:
                    continue

                if (next_row, next_col) == (7, 7):
                    # if reach the goal
                    reward = 1.0
                else:
                    # not the goal
                    reward = 0.0
                next_value = np.max(action_value[next_row, next_col, :])
                action_transform_score[_action] = transfer_function[_row, _col, _action] * (reward + gamma * next_value)

            best_action = int(action_transform_score.argmax(axis=0))
            new_pi[_row, _col, :] = 0.0
            new_pi[_row, _col, best_action] = 1.0

    return new_pi


ENVIRONMENT_SEED = 1000
MAPS_DEFINITION_TEMPLATE = [
    "FFFFHHFF",
    "FHFFFHFH",
    "FHFFFHFF",
    "FFFFFFFH",
    "FHHFFHFF",
    "FHFFFHHF",
    "FFFHFFFF",
    "HFFHHFFG",
]

PREDICTION_THRESHOLD = 1e-8
GAMMA = 0.9

# action-value function
action_value = np.ones([8, 8, 4]) * 0.25

# policy
pi = np.ones((8, 8, 4)) * 0.25

# initialize map
map = MAPS_DEFINITION_TEMPLATE
map[0] = "SFFFHHFF"

# initialize value of terminating state
for _row in range(len(map)):
    for _col in range(len(map[_row])):
        if map[_row][_col] == 'H':
            action_value[_row, _col] = 0.0
        elif map[_row][_col] == 'G':
            action_value[_row, _col] = 0.0

with Environment(map=map) as environment:
    environment.reset(seed=ENVIRONMENT_SEED)
    environment.render()

    epoch_delta = PREDICTION_THRESHOLD
    epoch_policy = [pi]
    epoch_values = [action_value]

    # Let It Begin!
    while epoch_delta >= PREDICTION_THRESHOLD:
        # 1. Policy Evaluation
        # 1.1 iterative updates action value within epoch
        action_values = [action_value]
        delta = PREDICTION_THRESHOLD
        while delta >= PREDICTION_THRESHOLD:
            _action_value = predict(map, pi, action_values[-1], gamma=GAMMA)

            # 1.2 Update Iteration Data
            action_values += [_action_value]
            delta = ((action_values[-1] - action_values[-2]) ** 2).sum()

        # 1.3 Print Action Value
        print(f'===================== epoch: {len(epoch_policy)} ---- converaged after {len(action_values)} iterations =====================')
        for _line in _action_value.tolist():
            print(list(__builtins__.map(lambda x: str(int(max(x) * 1000000) / 10000) + '%', _line)))

        # 2. Policy Improvement
        # 2.1 Update Pi Based on action-value Function
        pi = improve(map, pi, action_values[-1])

        # 2.2 Print New Policy
        policy_map = ['←', '↓', '→', '↑']
        print(f'===================== epoch: {len(epoch_policy)} ---- policy updated =====================')
        for _row in range(len(pi)):
            for _col in range(len(pi[_row])):
                best_action = int(pi[_row, _col].argmax(axis=0))
                print(policy_map[best_action], end=' ')
            print()

        # 3. Update Epoch Data
        epoch_policy += [pi]
        epoch_values += [_action_value]
        epoch_delta = ((epoch_values[-1] - epoch_values[-2]) ** 2).sum()
