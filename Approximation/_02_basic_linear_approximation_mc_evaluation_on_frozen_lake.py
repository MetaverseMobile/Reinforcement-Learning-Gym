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
    def __init__(self, input_size, output_size, zero_initalization=False, activation=None):
        # hidden layer: matrix of dimension (input_size, output_size)
        self.weights = np.asmatrix(np.random.randn(output_size, input_size))

        # bias: column vector of dimension (1, output_size)
        self.bias = np.asmatrix(np.random.randn(output_size, 1))

        if zero_initalization:
            self.weights *= 0.0
            self.bias *= 0.0

        # activation funcation
        if activation is not None:
            self.activation = activation
        else:
            self.activation = self.linear

    def linear(self, x):
        return x

    def set_zero(self):
        self.weights *= 0
        self.bias *= 0

    def softmax(self, x):
        exponents = np.exp(x)
        return exponents / np.sum(exponents)

    def forward(self, x):
        return self.weights.dot(np.asmatrix(x)) + self.bias

    def backpropagation(self, x, target, alpha=1e-3):
        x = np.asmatrix(x)

        delta = (self.weights.dot(x) + self.bias - target) / x.shape[-1]

        self.bias = self.bias - alpha * delta
        self.weights = self.weights - alpha * delta.dot(x.T)

    def reference(self, x):
        return self.activation(self.forward(x))

    def __call__(self, x):
        return self.reference(x)

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
            perception.backpropagation(x, G, alpha=1e-3)
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


def evaluate(base_map, pi, state_value_funcation, start_point=None, gamma=0.9, loops=1000, *args, **kwargs):

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
                action = pi[old_state].argmax(axis=0)

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

    Gs = dict()
    for _state, _Gs in returns.items():
        if len(_Gs):
            Gs[_state] = sum(_Gs) / len(_Gs)

    for _state, _G in Gs.items():
        state_one_hot = np.zeros([len(environment.map) * len(environment.map[0]), 1])
        state_one_hot[_state, 0] = 1.0

        estimated = state_value_funcation(state_one_hot)
        print(f"Loss {np.abs([[_G]] - estimated)} ---- Estimated value: {estimated} --- for position {_state} --- G: {_G}")
        state_value_funcation.backpropagation(state_one_hot, [[_G]])

    return state_value_funcation


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

# best policy
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

# value function
state_value_funcation = Perceptron(len(pi), 1, zero_initalization=1)

# roughly guess random value state table
positions_one_hot = np.zeros([len(pi), len(pi)])
for i in range(len(pi)):
    positions_one_hot[i, i] = 1.0

trainning_history = [state_value_funcation.forward(positions_one_hot)]

# iterate over interation with the environment
PREDICTION_THRESHOLD = 1e-8
MSE_loss = PREDICTION_THRESHOLD
while MSE_loss >= PREDICTION_THRESHOLD:
    state_value_funcation = evaluate(MAPS_DEFINITION_TEMPLATE, pi, state_value_funcation=state_value_funcation, loops=100)

    values = state_value_funcation(positions_one_hot)
    MSE_loss = np.sum(np.square(values - trainning_history[-1])) / len(pi)
    print(f"========================== epoch: {len(trainning_history)} ----- MSE Loss: {MSE_loss} ==========================")

    trainning_history += [values]

values = state_value_funcation(positions_one_hot)
print(f"========================== value funcation converged after {len(trainning_history)-1} epochs ----- current MSE Loss: {MSE_loss} ==========================")
print(values)
