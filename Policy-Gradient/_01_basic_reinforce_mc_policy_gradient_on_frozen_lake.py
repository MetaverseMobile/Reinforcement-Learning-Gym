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


class Perceptron(object):
    def __init__(self, input_size, output_size, state_space, action_space, zero_initalization=False, activation=None):
        # hidden layer: matrix of dimension (input_size, output_size)
        self.weights = np.asmatrix(np.random.randn(output_size, input_size))

        self.action_space = action_space
        self.inputs = np.asmatrix(np.eye(input_size))

        if zero_initalization:
            self.weights *= 0.0

    def set_zero(self):
        self.weights *= 0

    def inference(self, x):
        x = np.asmatrix(x)

        h = self.weights.dot(x.T).T
        t = h - np.max(h)
        h_s_a = np.exp(t)
        h_s = np.sum(h_s_a, axis=1)

        return (h_s_a / h_s)

    def __call__(self, x):
        return self.inference(x)

    def iterate(self, x, action, target, gamma_pow, alpha=1e-3):
        x = np.asmatrix(x)
        pmf = self.inference(x)

        grad = np.outer(np.eye(len(self.action_space))[action] - pmf, x)
        self.weights += alpha * gamma_pow * target * grad

    @staticmethod
    def linear_validation():
        x0 = np.linspace(-5, 5, 100)
        y0 = 5 * x0 + 4

        threshold = 1e-5

        x = x0.reshape(1, 100)
        G = y0.reshape(1, 100)
        perception = Perceptron(1, 1)

        max_epoch = 1000000
        for _epoch in range(max_epoch):
            res = perception.forward(x)
            error = np.square(res - G).sum() / x.shape[-1]
            perception.backpropagation(x, G, alpha=0.1)
            print(f"Epoch {_epoch + 1}: current MSE: {error}")

            if error < threshold:
                return True
        else:
            return False

    @staticmethod
    def validation():
        batch_size = 10000
        x = np.random.rand(64, batch_size)
        G = np.random.randn(4, batch_size) * 4  + 30

        threshold = 1e-5
        perception = Perceptron(64, 4)

        max_epoch = 1000000
        for _epoch in range(max_epoch):
            res = perception.forward(x)

            error = np.square(res - G).sum() / x.shape[-1]
            perception.backpropagation(x, G, alpha=1e-2)
            print(f"Epoch {_epoch + 1}: current MSE: {error}")

            if error < threshold:
                return True
        else:
            return False


def evaluate(base_map, policy_function, start_point=None, gamma=0.9, loops=1000, *args, **kwargs):
    _spots = [i for i, spot in enumerate(''.join(base_map)) if spot not in ('H', 'G')]

    inputs = np.eye(len(state_space))

    _loop = 0
    while _loop < loops:
        if start_point is None:
            _start_point = random.choice(_spots)
        else:
            _start_point = start_point

        _start_row = _start_point // len(base_map)
        _start_col = _start_point % len(base_map[0])
        generated_map = base_map[:_start_row] +                                                 \
            [base_map[_start_row][:_start_col] + 'S' + base_map[_start_row][_start_col+1:]] +   \
            base_map[_start_row+1:]

        updated = False
        with Environment(map=generated_map) as environment:
            old_state, info = environment.reset(seed=ENVIRONMENT_SEED)

            trajectory = list()
            first_visit = dict()

            episode_over = False
            while not episode_over:
                _input = inputs[old_state]
                # action = policy_function(_input).argmax(axis=1)[0, 0]
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
        returns = []
        for _idx, _item in reversed(list(enumerate(trajectory))):
            G = _item['reward'] + gamma * G

            _current_position = _item['from']

            # if first_visit[_current_position] == _idx:
            #     returns += [(_current_position, G, _item['action'])]
            returns += [(_idx, _current_position, G, _item['action'])]

        for timestamp, _state, _G, _action in returns:
            if _G > 0:
                updated = True
            _input = inputs[_state, ]
            estimated = policy_function(_input)
            print(f"Estimated policy: {estimated[0][0]} --- for position {_state} --- G: {_G}")
            policy_function.iterate(_input, _action, _G, gamma ** timestamp, alpha=0.1)

        if updated:
            _loop += 1

    return policy_function


# Environment settings
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

state_space = tuple(range(len(MAPS_DEFINITION_TEMPLATE) * len(MAPS_DEFINITION_TEMPLATE[0])))
action_space = (0, 1, 2, 3)

# policy function
policy_function = Perceptron(len(state_space), len(action_space), state_space, action_space, zero_initalization=True)

inputs = np.eye(len(state_space))
_output = policy_function(inputs)
policy_history = [_output]

# iterate over interation with the environment
PREDICTION_THRESHOLD = 1e-8
MSE_loss = PREDICTION_THRESHOLD
while MSE_loss >= PREDICTION_THRESHOLD:
    policy_function = evaluate(MAPS_DEFINITION_TEMPLATE, policy_function, loops=int(5e4))

    outputs = []
    for _state in state_space:
        _input = inputs[_state, ]
        _output = policy_function(_input)

        _output = [_item for _item in _output.tolist()[0]]
        outputs += [_output]
        print(_output)
    new_policy = np.array(outputs)

    MSE_loss = np.sum(np.square(new_policy - policy_history[-1]))

    policy_history += [new_policy]

    policy_map = ['←', '↓', '→', '↑']
    print(f"========================== epoch: {len(policy_history)} ----- MSE Loss: {MSE_loss} ==========================")
    for _row in range(len(new_policy)):
        best_action = int(new_policy[_row, :].argmax(axis=0))
        print(policy_map[best_action], end=' ')
        if _row % 8 == 7:
            print()

print(f"========================== value function converged after {len(policy_history)-1} epochs ----- current MSE Loss: {MSE_loss} ==========================")
