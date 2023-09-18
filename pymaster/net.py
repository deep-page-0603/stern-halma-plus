
from config import *
from util import *
from koni import *
from model import *
import numpy as np
import torch
import torch.nn as nn

class Net(object):
    def __init__(self, model_path = None, device = None):
        if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'        
        
        self.device = torch.device(device)
        self.model = Model().to(device = self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = init_lr, weight_decay = l2_const)

        if model_path is None:
            self.model_path = 'scratch'
        else:
            self.model_path = model_path
            self.load(model_path)

        self.model = nn.DataParallel(self.model)
        print('* loaded neural net from {} on {}'.format(self.model_path, device))

    def eval(self, states):
        self.model.eval()

        input_batches = self._split_batches_(states)
        batch_pi, batch_v = [], []

        for batch in input_batches:
            pi, v = self.model(self._parse_data_(batch, False))
            pi = torch.exp(pi)

            batch_pi.append(pi.data.cpu().numpy())
            batch_v.append(v.data.cpu().numpy())

        return np.vstack(batch_pi), np.hstack(batch_v)

    def train(self, data):
        self.model.train()        

        input_batches = self._split_batches_(data)
        sum_pi_loss, sum_v_loss = 0, 0
        
        for batch in input_batches:
            self.optimizer.zero_grad()
            
            states, true_pi, true_v = self._parse_data_(batch, True)
            pi, v = self.model(states)

            pi_loss = -torch.mean(torch.sum(pi * true_pi, (2, 1)))
            v_loss = F.mse_loss(v, true_v)

            sum_pi_loss += pi_loss.data.cpu().numpy()
            sum_v_loss += v_loss.data.cpu().numpy()

            loss = pi_loss + v_loss
            loss.backward()

            self.optimizer.step()

        pi_loss = sum_pi_loss / len(input_batches)
        v_loss = sum_v_loss / len(input_batches)

        return pi_loss, v_loss

    def save(self, model_path, only_model = False):
        save_dict = {
            'model': self.model.module.state_dict()
        }
        save_dict['optimizer'] = None if only_model else self.optimizer.state_dict()
        torch.save(save_dict, model_path)

    def load(self, model_path):
        ckpt = torch.load(model_path, map_location = self.device)
        self.model.load_state_dict(ckpt['model'])

        state_dict = ckpt['optimizer']
        if state_dict is not None: self.optimizer.load_state_dict(state_dict)

    def get_param_count(self):
        return sum(param.numel() for param in self.model.parameters())

    def get_model_path(self):
        return self.model_path

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _split_batches_(self, data):
        i, batches = 0, []

        if len(data) > batch_size:
            while i < len(data):
                batches.append(data[i:i + batch_size])
                i += batch_size
        else:
            batches.append(data)

        return batches

    def _parse_data_(self, data, has_true):
        if has_true:
            states, true_pi, true_v = [], [], []

            for s, pi, v in data:
                states.append(s)
                true_pi.append(pi)
                true_v.append(v)

            states = torch.from_numpy(np.stack(states)).float().to(self.device)
            true_pi = torch.from_numpy(np.stack(true_pi)).float().to(self.device)
            true_v = torch.from_numpy(np.array(true_v)).float().to(self.device)

            return states, true_pi, true_v
        else:
            states = torch.from_numpy(np.stack(data)).float().to(self.device)
            return states

if __name__ == '__main__':
    mkdir('./logs')

    net = Net()
    net.save('./logs/best.ckpt', True)

    params = net.get_param_count()
    print()
    
    print('params : {}'.format(params))
    print('size : {:.2f}mb'.format(params * 4 / 1024 / 1024))
