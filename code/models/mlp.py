import torch.nn as nn
# import math
import torch
from models import register
import pdb
import numpy as np

@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act='sine'):
        super().__init__()
        # pdb.set_trace()
        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU() 
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'sine':
            self.act = Siren()
        else:
            assert False, f'activation {act} is not supported'
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        if act is not None and act.lower() == 'sine':
            self.layers.apply(sine_init)
            self.layers[0].apply(first_layer_sine_init)

    def forward(self, x):
        # pdb.set_trace()
        shape = x.shape[:-1]
        x = self.layers(x.contiguous().view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class Siren(nn.Module):
    """
        Siren activation
        https://arxiv.org/abs/2006.09661
    """

    def __init__(self, w0=30):
        """
            w0 comes from the end of section 3
            it should be 30 for the first layer
            and 1 for the rest
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def extra_repr(self):
        return "w0={}".format(self.w0)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('first_layer_sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def init_weights(m):
    # if hasattr(modules, 'weight'):
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        # See supplement Sec. 1.5 for discussion of factor 30
        m.weight.data.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
