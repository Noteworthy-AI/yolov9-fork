import os
import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    # for v in model.modules():
    #    for p_name, p in v.named_parameters(recurse=0):
    #        if p_name == 'bias':  # bias (no decay)
    #            g[2].append(p)
    #        elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
    #            g[1].append(p)
    #        else:
    #            g[0].append(p)  # weight (with decay)

    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                g[1].append(v.im.implicit)
            else:
                for iv in v.im:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                g[1].append(v.ia.implicit)
            else:
                for iv in v.ia:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im2'):
            if hasattr(v.im2, 'implicit'):
                g[1].append(v.im2.implicit)
            else:
                for iv in v.im2:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia2'):
            if hasattr(v.ia2, 'implicit'):
                g[1].append(v.ia2.implicit)
            else:
                for iv in v.ia2:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im3'):
            if hasattr(v.im3, 'implicit'):
                g[1].append(v.im3.implicit)
            else:
                for iv in v.im3:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia3'):
            if hasattr(v.ia3, 'implicit'):
                g[1].append(v.ia3.implicit)
            else:
                for iv in v.ia3:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im4'):
            if hasattr(v.im4, 'implicit'):
                g[1].append(v.im4.implicit)
            else:
                for iv in v.im4:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia4'):
            if hasattr(v.ia4, 'implicit'):
                g[1].append(v.ia4.implicit)
            else:
                for iv in v.ia4:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im5'):
            if hasattr(v.im5, 'implicit'):
                g[1].append(v.im5.implicit)
            else:
                for iv in v.im5:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia5'):
            if hasattr(v.ia5, 'implicit'):
                g[1].append(v.ia5.implicit)
            else:
                for iv in v.ia5:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im6'):
            if hasattr(v.im6, 'implicit'):
                g[1].append(v.im6.implicit)
            else:
                for iv in v.im6:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia6'):
            if hasattr(v.ia6, 'implicit'):
                g[1].append(v.ia6.implicit)
            else:
                for iv in v.ia6:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im7'):
            if hasattr(v.im7, 'implicit'):
                g[1].append(v.im7.implicit)
            else:
                for iv in v.im7:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia7'):
            if hasattr(v.ia7, 'implicit'):
                g[1].append(v.ia7.implicit)
            else:
                for iv in v.ia7:
                    g[1].append(iv.implicit)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0, amsgrad=True)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    elif name == 'LION':
        optimizer = Lion(g[2], lr=lr, betas=(momentum, 0.99), weight_decay=0.0)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
