import numpy as np
import torch.nn as nn
import torch.distributions as distr

class SimpleConvNet(nn.Module):
    def __init__(self, side, nc, nf, nz):
        '''Simple CNN model for dataset

        Args:
            side: side size (such that image is side x side)
            nc: number of channes in image
            nf: number of features (scaled) after each convolution
            nz: number of features in last layer before fully connected layers
        '''

        super().__init__()
        
        self.nz = nz
        self.side = side
        
        # Number of layers
        l = int(np.floor(np.log2(side)))
        self.l = l
        
        # sizes = [nf * 2 ** (i - 1) for i in range(l + 1)][::-1]
        sizes = [nf * 2 ** (i - 1) for i in range(l + 1)]
        sizes[0] = nc
        
        layers = []
        for i in range(l):
            layers.append(nn.Conv2d(sizes[i], sizes[i + 1], 3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
        self.base = nn.Sequential(*layers)

        self.fc = nn.Sequential(nn.Linear(sizes[-1], nz), nn.Softmax(dim=1))
    

    def forward(self, x):
        b_size = x.size(0)
        z = self.base(x).reshape(b_size, -1)
        y_hat = self.fc(z)
        return y_hat

class AgentPolicyModel(nn.Module):
    def __init__(self, side, nc, nf, nz, feature_extractor=None):
        '''Simple CNN model

        Args:
            side: side size (such that image is side x side)
            nc: number of channes in image
            nf: number of features (scaled) after each convolution
            nz: number of features in last layer before fully connected layers
        '''
        
        super().__init__()

        if feature_extractor != None:
            self.base = feature_extractor
        else:
            self.base = SimpleConvNet(side, nc, nf, nz).base
        
        self.move_mean = nn.Sequential(nn.Linear(nz, 3), nn.Tanh())
        self.move_std = nn.Sequential(nn.Linear(nz, 3), nn.Softplus())
        
        self.guess_mean = nn.Sequential(nn.Linear(nz, 10), nn.Tanh())
        self.guess_std = nn.Sequential(nn.Linear(nz, 10), nn.Softplus())
    
    def forward(self, state):
        b_size = state.size(0)
        z = self.base(state).reshape(b_size, self.nz)
        
        move = (self.move_mean(z), self.move_std(z))
        guess = (self.guess_mean(z), self.guess_std(z))
        
        action = (move, guess)
        
        return action

class Reinforce(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.model = model
        self.onpolicy_reset()
        self.train()
        
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def forward(self, x):
        pdparams = self.model(x)
        return pdparams
    
    def act(self, state):
        pdparams = self.forward(state)
        
        pdmove_params = pdparams[0]
        pdmove = distr.normal.Normal(pdmove_params[0], pdmove_params[1])
        move = pdmove.sample()
        move_log_prob = pdmove.log_prob(move).sum()
#         print("move:", move)
        
#         pdguess = distr.Categorical(logits=pdparams[1])
#         guess = pdguess.sample()
#         guess_log_prob = pdguess.log_prob(guess)
        
        pdguess_params = pdparams[1]
        pdguess = distr.normal.Normal(pdguess_params[0], pdguess_params[1])
        guess = pdguess.sample()
#         print("guess prev:", guess)
        guess_log_prob = pdguess.log_prob(guess).sum()
        guess = nn.Softmax()(guess)
#         print("guess:", guess)
        
        
        
#         exp_guess_logits = torch.exp(guess_logits)
#         guess = exp_guess_logits / exp_guess_logits.sum()
#         guess_log_prob = torch.log(guess)
#         print("SHAPE:", guess_log_prob.shape)
        
        log_prob = move_log_prob + guess_log_prob # Equivalent to log(move_prob * guess_prob)
#         print("move_log_prob:", move_log_prob)
#         print("guess_log_prob:", guess_log_prob)
        self.log_probs.append(log_prob)
        
        action = [move.detach()[0], guess.detach()[0]]
        return action