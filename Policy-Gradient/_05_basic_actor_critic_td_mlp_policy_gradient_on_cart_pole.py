import gymnasium as gym
import numpy as np
from collections import deque
from dataclasses import dataclass
import random
import copy


np.set_printoptions(threshold=400, edgeitems=12)

# Environment settings
ENVIRONMENT_SEED = 1000

# Training settings
TUNING = False


class Environment(object):
    def __init__(self, render_mode=None, *args, **kwargs):
        # Initialise the environment
        self.environment = gym.make(
            'CartPole-v1',
            render_mode=render_mode,
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


class Perceptron(object):
    def __init__(self, input_size, output_size, zero_initalization=False):
        # hidden layer: matrix of dimension (input_size, output_size)
        self.actor_weights = [
            np.asmatrix(np.random.randn(16, input_size)) * 0.01,
            np.asmatrix(np.random.randn(24, 16))  * 0.01,
            np.asmatrix(np.random.randn(output_size, 24))  * 0.01,
        ]

        self.actor_bias = [
            np.asmatrix(np.random.randn(16, 1)) * 0.01,
            np.asmatrix(np.random.randn(24, 1))  * 0.01,
            np.asmatrix(np.random.randn(2, 1))  * 0.01,
        ]

        self.critic_weights = [
            np.asmatrix(np.random.randn(16, input_size))  * 0.01,
            np.asmatrix(np.random.randn(1, 16))  * 0.01,
        ]

        self.critic_bias = [
            np.asmatrix(np.random.randn(16, 1))  * 0.01,
            np.asmatrix(np.random.randn(1, 1))  * 0.01,
        ]

        self.target_weights = copy.deepcopy(self.critic_weights)
        self.target_bias = copy.deepcopy(self.critic_bias)

        if zero_initalization:
            self.set_zero()

    def set_zero(self):
        for _layer in self.actor_weights \
                    + self.actor_bias \
                    + self.critic_weights \
                    + self.critic_bias \
                    + self.target_weights \
                    + self.target_bias:
            _layer *= 0

    def actor(self, x, hidden_layer_output=False):
        x = np.asmatrix(x)

        # forward propagation
        hidden_output1 = self.actor_weights[0].dot(x.T) + self.actor_bias[0]
        activation1 = np.tanh(hidden_output1)
        hidden_output2 = self.actor_weights[1].dot(activation1)  + self.actor_bias[1]
        activation2 = np.tanh(hidden_output2)
        hidden_output3 = self.actor_weights[2].dot(activation2) + self.actor_bias[2]

        h = hidden_output3.T
        t = h - np.max(h, axis=1)
        h_s_a = np.exp(t)
        h_s = np.sum(h_s_a, axis=1)

        if hidden_layer_output:
            return (h_s_a / h_s), [hidden_output1.T, hidden_output2.T, hidden_output3.T]
        else:
            return (h_s_a / h_s)

    def critic(self, x, hidden_layer_output=False):
        x = np.asmatrix(x)

         # forward propagation
        hidden_output1 = self.critic_weights[0].dot(x.T) + self.critic_bias[0]
        activation1 = np.tanh(hidden_output1)

        hidden_output2 = self.critic_weights[1].dot(activation1) + self.critic_bias[1]

        if hidden_layer_output:
            return hidden_output2.T, [hidden_output1.T, hidden_output2.T]
        else:
            return hidden_output2.T

    def target(self, x, hidden_layer_output=False):
        x = np.asmatrix(x)

        # forward propagation
        hidden_output1 = self.target_weights[0].dot(x.T) + self.target_bias[0]
        activation1 = np.tanh(hidden_output1)
        hidden_output2 = self.target_weights[1].dot(activation1)  + self.target_bias[1]

        if hidden_layer_output:
            return hidden_output2.T, [hidden_output1.T, hidden_output2.T]
        else:
            return hidden_output2.T

    def update_target(self, tau=0.1):
        for i in range(len(self.critic_weights)):
            self.target_weights[i] = (1.0 - tau) * self.target_weights[i] + tau * self.critic_weights[i]
            self.target_bias[i] = (1.0 - tau) * self.target_bias[i] + tau * self.critic_bias[i]

    def __call__(self, x):
        return self.actor(x)

    def iterate(self, x, action, reward, new_state, terminated=False, gamma=0.9, actor_alpha=1e-3, critic_alpha=1e-3):
        x = np.asmatrix(x)

        pmf, policy_hidden_layer_outputs = self.actor(x, hidden_layer_output=True)
        current_value, value_hidden_layer_outputs = self.critic(x, hidden_layer_output=True)

        new_value = np.multiply((1 - terminated), self.target(new_state))
        delta = reward + gamma * new_value - current_value

        grad3 = np.multiply(delta, np.eye(2)[action].reshape(-1, 2) - pmf)
        grad2 = np.multiply(grad3.dot(self.actor_weights[2]), (1 - np.square(np.tanh(policy_hidden_layer_outputs[1]))))
        grad1 = np.multiply(grad2.dot(self.actor_weights[1]), (1 - np.square(np.tanh(policy_hidden_layer_outputs[0]))))

        self.actor_weights[2] += actor_alpha * grad3.T.dot(policy_hidden_layer_outputs[1])
        self.actor_weights[1] += actor_alpha * grad2.T.dot(policy_hidden_layer_outputs[0])
        self.actor_weights[0] += actor_alpha * grad1.T.dot(x)

        self.actor_bias[1] += actor_alpha * grad2.mean(axis=0).T
        self.actor_bias[0] += actor_alpha * grad1.mean(axis=0).T

        grad_v2 = delta
        grad_v1 = np.multiply(grad_v2.dot(self.critic_weights[1]), (1 - np.square(np.tanh(value_hidden_layer_outputs[0]))))

        self.critic_weights[1] += critic_alpha * grad_v2.T.dot(value_hidden_layer_outputs[0])
        self.critic_weights[0] += critic_alpha * grad_v1.T.dot(x)

        self.critic_bias[1] += critic_alpha * grad_v2.mean(axis=0).T
        self.critic_bias[0] += critic_alpha * grad_v1.mean(axis=0).T

        return delta


def evaluate(policy_function, start_point=None, gamma=0.9, loops=1000, actor_alpha=1e-3, critic_alpha=1e-3, render_mode=None, *args, **kwargs):
    td_errors = []
    accumulations = []
    with Environment(render_mode=render_mode) as environment:
        for _loop in range(loops):
            accumulation = 0
            old_state, info = environment.reset()
            old_state[2] = np.tan(old_state[2])

            episode_over = False
            while not episode_over:
                action = np.random.choice([0, 1], p=np.array(policy_function(old_state)).flatten())

                # observation
                # 1. Cart Position         |        -4.8         |       4.8
                # 2. Cart Velocity         |        -Inf         |       Inf
                # 3. Pole Angle            | ~ -0.418 rad (-24°) |  ~ 0.418 rad (24°)
                # 4. Pole Angular Velocity |        -Inf         |       Inf
                _new_state, reward, terminated, truncated, info = environment.step(action)
                _new_state[2] = np.tan(_new_state[2])
                new_state = np.asmatrix(_new_state)
                accumulation += reward

                delta = policy_function.iterate(old_state, action, reward, new_state, terminated=terminated, gamma=gamma, actor_alpha=actor_alpha, critic_alpha=critic_alpha)

                old_state = new_state
                episode_over = terminated or truncated
                td_errors += [np.mean(delta)]
        accumulations += [accumulation]

    return policy_function, accumulations, td_errors


# policy function
policy_function = Perceptron(4, 2)

sequence_inputs = np.mgrid[-4.8:4.8:10j, -100:100:5j, -0.5:0.5:5j, -100:100:5j].reshape(-1, 4)

REWARD_THRESHOLD = 500
average_reward = 0

GAMMA = 0.99
ACTOR_ALPHA = 1e-4
CRITIC_ALPHA = 3e-4
LOOPS = int(3e1)
RENDER = None

epoch = 0
while average_reward < REWARD_THRESHOLD:
    policy_function, accumulations, td_errors = evaluate(policy_function, loops=LOOPS, gamma=GAMMA, actor_alpha=ACTOR_ALPHA, critic_alpha=CRITIC_ALPHA, render_mode=RENDER)
    average_reward = int(np.mean(accumulations))

    epoch += 1
    print(f'====================== epoch: {epoch} ----- average td error: {np.mean(td_errors)} ----- accumulated: {average_reward} ======================')

    policy_function.update_target()

    RENDER = None
    if TUNING:
        values = policy_function.critic(sequence_inputs)
        print(f'{values.T} --------------- {np.mean(values)}')
        actions = policy_function(sequence_inputs)
        print(f'{actions}')
        pos_count = actions.argmax(axis=1).sum()
        print(f"Pos-Neg balance: {pos_count} ----- {len(actions) - pos_count}")
        average_reward = 0 # in tuning mode, we train the agent repeatly, until everything is ready and debug mode is disabled
