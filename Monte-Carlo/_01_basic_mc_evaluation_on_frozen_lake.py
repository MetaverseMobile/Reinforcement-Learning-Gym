import gymnasium as gym
import numpy as np
import random


class Environment(object):
    def __init__(self, map=None, is_slippery=False, *args, **kwargs):
        # Initialise the environment
        self.environment = gym.make(
            'FrozenLake-v1',
            desc=map,
            is_slippery=is_slippery,
            render_mode="human",
        )
        self.map = map

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


def evaluate(base_map, pi, start_point=None, gamma=0.9, loops=1000, *args, **kwargs):
    trajectories = []

    _spots = [i for i, spot in enumerate(''.join(base_map)) if spot not in ('H', 'G')]
    returns = {_spot: list() for _spot in _spots}

    for _loop in range(loops):
        print(f'++++++++++++++++++++++++ loop: {_loop+1} ++++++++++++++++++++++++')
        if start_point is None:
            _start_point = random.choice(_spots)
        else:
            _start_point = start_point

        _start_row = _start_point // len(base_map)
        _start_col = _start_point % len(base_map[0])
        generated_map = base_map[:_start_row] +                                                 \
            [base_map[_start_row][:_start_col] + 'S' + base_map[_start_row][_start_col+1:]] +   \
            base_map[_start_row+1:]

        with Environment(map=generated_map) as environment:
            old_state, info = environment.reset(seed=ENVIRONMENT_SEED)

            trajectory = list()
            first_visit = dict()

            episode_over = False
            while not episode_over:
                action = environment.sample_action_space()

                new_state, reward, terminated, truncated, info = environment.step(action)
                trajectory += [{
                    "from": old_state,
                    "action": int(action),
                    "reward": reward,
                    "to": new_state,
                }]

                if old_state not in first_visit:
                    first_visit[old_state] = len(trajectory) - 1

                old_state = new_state
                episode_over = terminated or truncated

        # for one trajectory
        G = 0
        for _idx, _item in reversed(list(enumerate(trajectory))):
            G = _item['reward'] + gamma * G

            _current_position = _item['from']
            if first_visit[_current_position] == _idx:
                returns[_current_position] += [G]

        trajectories += [trajectory]

        print(f"current returns: {returns}")
    return returns, trajectories


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

returns = { state: list() for state in range(len(MAPS_DEFINITION_TEMPLATE) * len(MAPS_DEFINITION_TEMPLATE[0]))}

# policy
pi = np.array([
    [0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.25, 0.25, 0.25, 0.25],[0.0, 1.0, 0.0, 0.0],[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],
    [0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.0, 1.0, 0.0, 0.0],[1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],
    [0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.25, 0.25, 0.25, 0.25],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.25, 0.25, 0.25, 0.25],[0.25, 0.25, 0.25, 0.25],[0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0],[0.25, 0.25, 0.25, 0.25],[0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],
    [0.25, 0.25, 0.25, 0.25],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0],[0.25, 0.25, 0.25, 0.25],[0.25, 0.25, 0.25, 0.25],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.25, 0.25, 0.25, 0.25],
])

# Policy Evaluation
PREDICTION_THRESHOLD = 1e-8
delta = PREDICTION_THRESHOLD
state_values = []
while delta >= PREDICTION_THRESHOLD:
    _returns, trajectories = evaluate(MAPS_DEFINITION_TEMPLATE, pi, loops=int(64e2))

    for _state, G in _returns.items():
        returns[_state] += G

    estimated_values = dict()
    for state, Gs in returns.items():
        if len(Gs) != 0:
            estimated_values[state] = sum(Gs) / len(Gs)
        else:
            estimated_values[state] = 0.0

    print(f"============================== epoch: {len(state_values)+1} --- delta: {delta} ==============================")
    for key, values in estimated_values.items():
        print(f"{key}: {values}")

    if state_values and all(estimated_values.values()):
        _delta = 0
        for _state in range(len(MAPS_DEFINITION_TEMPLATE) * len(MAPS_DEFINITION_TEMPLATE[0])):
            _delta += (estimated_values[_state] - state_values[-1][_state]) ** 2

        delta = _delta

    state_values += [estimated_values]
