import gymnasium as gym
import numpy as np
import copy
import random

# Environment settings
ENVIRONMENT_SEED = 1000

# Training settings
TUNING = False


class Environment(object):
    def __init__(self, render_mode=None,*args, **kwargs):
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
    def __init__(self, input_size, output_size, zero_initalization=False, activation=None):
        # hidden layer: matrix of dimension (input_size, output_size)
        self.actor_weights = np.asmatrix(np.random.randn(output_size, input_size))
        self.actor_bias = np.asmatrix(np.random.randn(2))
        self.critic_weights = np.asmatrix(np.random.randn(1, input_size))
        self.critic_bias = np.asmatrix(np.random.randn(1))
        self.target_weights = copy.deepcopy(self.critic_weights)
        self.target_bias = copy.deepcopy(self.critic_bias)

        if zero_initalization:
            self.set_zero()

    def set_zero(self):
        self.actor_weights *= 0
        self.actor_bias *= 0
        self.critic_weights *= 0
        self.critic_bias *= 0
        self.target_weights *= 0
        self.target_bias *= 0

    def actor(self, x, hidden_layer_output=False):
        x = np.asmatrix(x)

        h = self.actor_weights.dot(x.T).T + self.actor_bias
        t = h - np.max(h)
        h_s_a = np.exp(t)
        h_s = np.sum(h_s_a, axis=1)

        if hidden_layer_output:
            return (h_s_a / h_s), h
        else:
            return h_s_a / h_s

    def __call__(self, x):
        return self.actor(x)

    def critic(self, x):
        x = np.asmatrix(x)
        return self.critic_weights.dot(x.T) + self.critic_bias

    def target(self, x):
        x = np.asmatrix(x)
        return self.target_weights.dot(x.T) + self.target_bias

    def iterate(self, x, action, reward, new_state, terminated=False, gamma=0.9, actor_alpha=1e-3, critic_alpha=1e-3):
        x = np.asmatrix(x)
        pmf, h = self.actor(x, hidden_layer_output=True)

        next_value = 0 if terminated else self.target(new_state)[0, 0]
        delta = reward + gamma * next_value - self.critic(x)[0, 0]

        grad = np.eye(2)[action].reshape(-1, 2) - pmf

        self.actor_weights += actor_alpha * delta * grad.T.dot(x)
        self.actor_bias += actor_alpha * delta
        self.critic_weights += critic_alpha * delta * x
        self.critic_bias += critic_alpha * delta

        return delta

    def update_target(self, tau=0.1):
        self.target_weights = tau * self.critic_weights + (1 - tau) * self.target_weights
        self.target_bias = tau * self.critic_bias + (1 - tau) * self.target_bias


def evaluate(policy_function, start_point=None, gamma=0.9, loops=1000, render_mode=None, actor_alpha=1e-4, critic_alpha=3e-4, *args, **kwargs):
    accumulations = []
    td_errors = []
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

                delta = policy_function.iterate(old_state, action, reward, new_state, terminated, gamma=gamma, actor_alpha=actor_alpha, critic_alpha=critic_alpha)

                old_state = new_state
                td_errors += [np.mean(delta)]
                episode_over = terminated or truncated
        accumulations += [accumulation]

    return policy_function, accumulations, td_errors


# policy function
policy_function = Perceptron(4, 2)

sequence_inputs = np.mgrid[-4.8:4.8:10j, -100:100:5j, -0.5:0.5:5j, -100:100:5j].reshape(-1, 4)

REWARD_THRESHOLD = 300
average_reward = 0

GAMMA = 0.99
ACTOR_ALPHA = 1e-6
CRITIC_ALPHA = 3e-6
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
        average_reward = 0 # in debug mode, we train the agent repeatly, until everything is ready and debug mode is disabled
