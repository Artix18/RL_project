import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
import ipdb
import numpy as np

#######################################################################
# We used the code: https://github.com/JamesChuanggg/pytorch-REINFORCE
# and modified it so as to make an actor-critic with epsilon greedy
# (ie we just kept the reinforce formula and the gradient clipping)
#######################################################################

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(18, 128)
        self.action_head = nn.Linear(128, 9)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values


class REINFORCE:
    def __init__(self):
        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, value_estim = self.model(Variable(state))       
        action = probs.multinomial().data
        alpha = np.random.rand()
        if(alpha < 1e-2):
            unif = Variable(torch.from_numpy(np.array([1./9]*9)).float().unsqueeze(0))
            action = unif.multinomial().data
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()

        return action[0], log_prob, entropy, value_estim

    def update_parameters(self, rewards, log_probs, entropies, vls, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        value_loss=0
        for i in reversed(range(len(rewards))):
            rewards[i] -= vls[i].data[0,0]
            R = gamma * R + rewards[i]
            value_loss += F.smooth_l1_loss(vls[i], Variable(torch.Tensor([rewards[i]])))
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() #- (0.000001*entropies[i]).sum()
        loss += value_loss
        loss = loss / len(rewards)
		
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
        
        return loss,value_loss
