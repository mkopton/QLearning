import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import SarsaAgent, QLearningAgent

env = gym.make('CliffWalking-v0')
# YOUR CODE HERE
# Set the parameters of the agent
sarsaAgent = SarsaAgent(env, alpha=0.5, gamma=0.95, epsilon=0.95)
env.reset()
sarsa_Rewards = []
# YOUR CODE HERE
# Change the number of episodes, if necessary
for episode in tqdm(range(200000)):
    episode_reward = 0
    state, info = env.reset()
    # Use the agent to choose the first action
    action = sarsaAgent.get_action(state) # YOUR CODE HERE
    done = False
    while not done:
        # Perform the selected action
        next_state, reward, done, truncated, info = env.step(action)
        # Select next action        
        next_action = sarsaAgent.get_action(next_state) # YOUR CODE HERE
        # Update the agent
        sarsaAgent.update(state, action, reward, next_state, next_action)
        
        # Assign the next state and action to the current ones
        state = next_state # YOUR CODE HERE
        action = next_action # YOUR CODE HERE
        episode_reward += reward
    sarsa_Rewards.append(episode_reward)

sarsaAgent.save()
plt.plot(sarsa_Rewards)

# YOUR CODE HERE
# Set the parameters of the agent
qAgent = QLearningAgent(env, alpha=0.5, gamma=0.95, epsilon=0.9)

Q_rewards = []
for episode in tqdm(range(2000)):
    episode_reward = 0
    state, info = env.reset()
    done = False
    while not done:
        # Choose an action
        action = qAgent.get_action(state) # YOUR CODE HERE
        
        # Perform the selected action
        next_state, reward, done, truncated, info = env.step(action) # YOUR CODE HERE

        # Update the agent
        # YOUR CODE HERE
        qAgent.update(state, action, reward, next_state)
        # Assign the next state to the current one
        state = next_state # YOUR CODE HERE
        
        episode_reward += reward
    Q_rewards.append(episode_reward)
        
qAgent.save()
plt.plot(Q_rewards)


env_taxi = gym.make('Taxi-v3')
# YOUR CODE HERE

sarsaAgent_taxi = SarsaAgent(env_taxi, alpha=0.5, gamma=0.95, epsilon=1)
env_taxi.reset()
sarsa_Rewards_taxi = []
# YOUR CODE HERE
# Change the number of episodes, if necessary
for episode in tqdm(range(150000)):
    episode_reward = 0
    state, info = env_taxi.reset()
    # Use the agent to choose the first action
    action = sarsaAgent_taxi.get_action(state) # YOUR CODE HERE
    done = False
    while not done:
        # Perform the selected action
        next_state, reward, done, truncated, info = env_taxi.step(action)
        # Select next action        
        next_action = sarsaAgent_taxi.get_action(next_state) # YOUR CODE HERE
        # Update the agent
        sarsaAgent_taxi.update(state, action, reward, next_state, next_action)
        
        # Assign the next state and action to the current ones
        state = next_state # YOUR CODE HERE
        action = next_action # YOUR CODE HERE
        episode_reward += reward
    sarsa_Rewards_taxi.append(episode_reward)

sarsaAgent_taxi.save('Taxi_Sarsa.npy')
plt.plot(sarsa_Rewards_taxi)


qAgent_taxi = QLearningAgent(env_taxi, alpha=0.5, gamma=0.95, epsilon=1)
env_taxi.reset()
qLearning_Rewards_taxi = []
# YOUR CODE HERE
# Change the number of episodes, if necessary
for episode in tqdm(range(50000)):
    episode_reward = 0
    state, info = env_taxi.reset()
    # Use the agent to choose the first action
    action =  qAgent_taxi.get_action(state) # YOUR CODE HERE
    done = False
    while not done:
        # Perform the selected action
        next_state, reward, done, truncated, info = env_taxi.step(action)
        # Select next action        
        next_action = qAgent_taxi.get_action(next_state) # YOUR CODE HERE
        # Update the agent
        qAgent_taxi.update(state, action, reward, next_state)
        
        # Assign the next state and action to the current ones
        state = next_state # YOUR CODE HERE
        action = next_action # YOUR CODE HERE
        episode_reward += reward
    qLearning_Rewards_taxi.append(episode_reward)

qAgent_taxi.save('Taxi_qLearn.npy')
plt.plot(sarsa_Rewards_taxi)