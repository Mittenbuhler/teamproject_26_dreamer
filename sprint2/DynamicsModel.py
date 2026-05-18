import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DynamicsModel(nn.Module):
  def __init__(self):
    super(DynamicsModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        self.s_head = nn.Linear(hidden_dim, state_dim)   
        self.r_head = nn.Linear(hidden_dim, 1)            
  
  def forward(self,s,a):
    s = F.relu(self.layer1(s))
    a = F.relu(self.layer2(a))
  

  def loss(s1,a,s2,r):
      #ToDo
  

   def collect_transitions(env, n_episodes):
    return 1
     #for episode in range(n_episodes):
       #make random experiences 𝑠%, 𝑎%, 𝑠%&', 𝑟 
    #return buffer (all experience)
   
    
def train(n_training_steps):
    return 1
    #ToDo
