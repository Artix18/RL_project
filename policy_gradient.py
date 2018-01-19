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

#loss
win_loss = viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

#rewards

win_rewards = viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

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

agent = REINFORCE()

running_reward = 0.5
disp_rewards = []
total_losses = []
value_losses = []
for i_episode in count(1):
    state = env.reset(np.random.randint(3))
    entropies = []
    log_probs = []
    rewards = []
    vls=[]
    for t in range(10000): # Don't infinite loop while learning
        action, log_prob, entropy, value_estim = agent.select_action(state)
        state, reward = env.play_action(action[0])

        #model.rewards.append(reward)
        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        vls.append(value_estim)
        
        if reward==1:
            state, ok = env.next_task()
            if ok==0:
                break

    #mean_action = np.mean(model.saved_actions)
    
    running_reward = running_reward * 0.99 + np.mean(rewards) * 0.01
    disp_rewards.append(running_reward)
    total_loss, value_loss = agent.update_parameters(rewards, log_probs, entropies, vls, args.gamma)
    #ipdb.set_trace()
    total_losses.append(total_loss.data.numpy()[0])
    value_losses.append(value_loss.data.numpy()[0])
    #finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}'.format(
            i_episode, t, running_reward))
    if i_episode % args.display_interval == 0:
        truc = np.zeros((3,9))
        probs= np.zeros((3,9))
        for st_id in range(3):
            env.reset(st_id)
            env.personnalite = env.mu[st_id,:]
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
        title='theta for each pair (student_type, action)'
        #colormap='Electric',
        ))
        viz.heatmap(X=probs,win=win_probs, opts=dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f','g','h','i'],
        rownames=['mu_1', 'mu_2', 'mu_3'],
        title='Probabilities (output of nn) for profile mu_i'
        #colormap='Electric',
        ))
        viz.line(X=np.arange(1,i_episode+1), Y=np.array(disp_rewards), win=win_rewards, opts=dict(title='Evolution of the running reward', xlabel='episode', ylabel='reward'))
        vl = np.array(value_losses)
        tl = np.array(total_losses)
        #ipdb.set_trace()
        my_y = np.array([vl, tl]).T
        viz.line(X=np.arange(1,i_episode+1), Y=my_y, win=win_loss, opts=dict(title='Evolution of the losses', legend=['value_loss','total_loss'],xlabel='episode', ylabel='loss'))
    if i_episode > 10 and running_reward > 0.95:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break
