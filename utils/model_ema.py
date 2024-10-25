import math
from copy import deepcopy
from .torch_utils import de_parallel, copy_attr


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def smart_resume(ckpt, optimizer, ema, epochs):
    # Resume training from a partially trained checkpoint
    best_fitness = 0.0
    start_epoch = 0
    if ckpt.get('epoch') and isinstance(ckpt.get('epoch'), int):
        start_epoch = ckpt['epoch'] + 1
    if optimizer and ckpt.get('optimizer') and ckpt.get('best_fitness'):
        optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
        best_fitness = ckpt['best_fitness']
    if ema and ckpt.get('ema') and ckpt.get('updates'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
        ema.updates = ckpt['updates']
    if epochs < start_epoch:
        assert start_epoch < epochs, f"Start epoch {start_epoch} from checkpoint cannot be higher than total epochs {epochs}."
    return best_fitness, start_epoch