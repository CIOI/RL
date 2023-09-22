# classical gym 
import gym
# instead of gym, import gymnasium 
#import gymnasium as gym
import numpy as np
import time
import torch

l1 = 4 #A
l2 = 150
l3 = 2 #B

def loss_fn(preds, r): #A
    return -1 * torch.sum(r * torch.log(preds)) #B
def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards #A
    disc_return /= disc_return.max() #B
    return disc_return
model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax(dim=0) #C
)

learning_rate = 0.009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# create environment
env=gym.make('CartPole-v1',render_mode='human')
# reset the environment, 
# returns an initial state
(state,_)=env.reset()
# states are
# cart position, cart velocity 
# pole angle, pole angular velocity

MAX_DUR = 200
MAX_EPISODES = 500
gamma = 0.99
score = [] #A
expectation = 0.0
for episode in range(MAX_EPISODES):
    curr_state = env.reset()[0]
    print(episode)
    env.render()
    transitions=[]
    for t in range(MAX_DUR):
        act_prob = model(torch.from_numpy(curr_state).float()) #D
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy()) #E
        prev_state = curr_state
        curr_state, _, done, info = env.step(action)[0:4] #F
        transitions.append((prev_state, action, t+1)) #G
        time.sleep(0.1)
        if (done):
            time.sleep(1)
            break
    ep_len = len(transitions) #I
    score.append(ep_len)
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,)) #J
    disc_returns = discount_rewards(reward_batch) #K
    state_batch = torch.Tensor([s for (s,a,r) in transitions]) #L
    action_batch = torch.Tensor([a for (s,a,r) in transitions]) #M
    pred_batch = model(state_batch) #N
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() #O
    loss = loss_fn(prob_batch, disc_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
env.close()   