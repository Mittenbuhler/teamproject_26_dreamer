import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DynamicsModel(nn.Module):
  def __init__(self):
    super(DynamicsModel, self).__init__()
    self.layer1 = nn.Linear(6, 64)
    self.layer2  = nn.Linear(64, 64)
    self.s_head = nn.Linear(64, 4)
    self.r_head = nn.Linear(64, 1)
  
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
