import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from src.env import ImageClassificationEnv
from src.model import AgentPolicyModel, Reinforce

data_path = '/Users/cesar.salcedo/Documents/datasets/mnist'
image_size = 16

env = ImageClassificationEnv(data_path, image_size, action_type='position')
model = AgentPolicyModel(image_size, 1, 16, 128)
agent = Reinforce(model)

gamma = 0.99
def train(agent, optimizer):
    T = len(agent.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = agent.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(agent.log_probs)
#     print(log_probs.shape)
#     print(rets.shape)
    loss = -log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

    
episodes = 60000
max_timesteps = 2000

optimizer = optim.Adam(agent.parameters(), lr=0.005)

for episode in tqdm(range(episodes)):
    state = env.reset()
    
    for t in range(max_timesteps):
        action = agent.act(state)
        # print("Action:", action)
        # print("Threshold:", env.threshold)
        state, reward, done = env.step(action)
        agent.rewards.append(reward)
#         env.render()
        if done:
            break
        
    
    loss = train(agent, optimizer)
    total_reward = sum(agent.rewards)
    agent.onpolicy_reset()
    print('Episode {}, loss: {}, total_reward: {}'.format(episode, loss, total_reward))