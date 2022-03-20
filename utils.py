import random
import numpy as np
import torch

class CofStepController:
    def __init__(self, init_cof=.5, gamma=5., milestones=[20, 100]):
        self.init_cof = init_cof
        self.cof = init_cof
        self.gamma = gamma
        self.milestones = np.array(milestones)
        self.count = 0
    
    def step(self):
        self.count += 1
        factor = self.gamma ** sum(self.milestones < self.count)
        self.cof = self.init_cof * factor
            
        
def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.determinstic = True
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)