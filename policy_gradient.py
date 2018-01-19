import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import env
from visdom import Visdom
from reinforce import REINFORCE

viz = Visdom()

# bar for thetas = truc^T bidule
win_theta = viz.heatmap(
    X=np.outer(np.arange(1,4), np.arange(1,10)),
    opts=dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f','g','h','i'],
        rownames=['type_1', 'type_2', 'type_3'],
        #colormap='Electric',
        )
    )
#viz.bar(X=np.random.rand(9))

# heatmap
opts_probs = dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f','g','h','i'],
        rownames=['type_1', 'type_2', 'type_3'],
        #colormap='Electric',
    )
win_probs = viz.heatmap(
    X=np.outer(np.arange(1, 4), np.arange(1, 10)),
    opts=opts_probs
)

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--display-interval', type=int, default=100)
args = parser.parse_args()


env = env.Env()
#env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


SavedAction = namedtuple('SavedAction', ['action', 'value'])
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(18, 128)
        self.action_head = nn.Linear(128, 9)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)


def select_action(state):
    #ipdb.set_trace()
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    action = probs.multinomial()
    alpha = np.random.rand()
    ipdb.set_trace()
    if(alpha < 0.01):
        #ipdb.set_trace()
        unif = Variable(torch.from_numpy(np.array([1./9]*9)).float().unsqueeze(0))
        action = unif.multinomial()
    model.saved_actions.append(SavedAction(action, state_value))
    return action.data


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    value_loss = 0
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (action, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0,0]
        action.reinforce(reward)
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r])))
    optimizer.zero_grad()
    final_nodes = [value_loss] + list(map(lambda p: p.action, saved_actions))
    gradients = [torch.ones(1)] + [None] * len(saved_actions)
    autograd.backward(final_nodes, gradients)
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

agent = REINFORCE()

running_reward = 0.5
for i_episode in count(1):
    state = env.reset(np.random.randint(3))
    entropies = []
    log_probs = []
    rewards = []
    for t in range(10000): # Don't infinite loop while learning
        action, log_prob, entropy = agent.select_action(state)
        state, reward = env.play_action(action[0])

        #model.rewards.append(reward)
        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        
        if reward==1:
            state, ok = env.next_task()
            if ok==0:
                break

    #mean_action = np.mean(model.saved_actions)
    
    running_reward = running_reward * 0.99 + np.mean(rewards) * 0.01
    agent.update_parameters(rewards, log_probs, entropies, args.gamma)
    #finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}'.format(
            i_episode, t, running_reward))
    if i_episode % args.display_interval == 0:
        truc = np.zeros((3,9))
        probs= np.zeros((3,9))
        for st_id in range(3):
            env.reset(st_id)
            #env.personnalite = env.mu[st_id,:]
            disp_state = env.get_state()
            disp_state = torch.from_numpy(disp_state).float().unsqueeze(0)
            my_probs, my_state_value = agent.model(Variable(disp_state))
            #ipdb.set_trace()
            for action in range(9):
                my_theta = np.abs(np.dot(env.actions_theta[action,:], env.mu[env.student_type, :]) / (np.linalg.norm(env.actions_theta[action,:]) * np.linalg.norm(env.mu[env.student_type,:])))
                #np.abs(np.dot(env.actions_theta[action,:], env.personnalite) / (np.linalg.norm(env.actions_theta[action,:]) * np.linalg.norm(env.personnalite)))
                probs[st_id, action]=my_probs.data.numpy()[0,action]
                truc[st_id, action]=my_theta
        viz.heatmap(X=truc, win=win_theta, opts=dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f','g','h','i'],
        rownames=['type_1', 'type_2', 'type_3'],
        #colormap='Electric',
        ))
        viz.heatmap(X=probs,win=win_probs, opts=dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f','g','h','i'],
        rownames=['type_1', 'type_2', 'type_3'],
        #colormap='Electric',
    ))
    if i_episode > 10 and running_reward > 0.95:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break
