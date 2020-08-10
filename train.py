from env.pid import PidEnv
import numpy as np
from ddpg import Agent
from OUNoise import Noise
import matplotlib.pyplot as plt

env = PidEnv(setpoint=20)
batch_size = 128
rewards = []
agent = Agent(num_states=5, num_actions=3)
noise = Noise(num_actions=3)

for episode in range(30):
    state = env.reset()
    noise.reset()
    eps_reward = 0
    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)

        new_state, reward = env.step(action)

        agent.mem.push(state, action, reward, new_state)

        agent.learn(batch_size)

        state = new_state

        eps_reward += reward
    rewards.append(eps_reward)


plt.plot(rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
