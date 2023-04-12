# Copyright (c) EEEM071, University of Surrey
import math
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from solver.cosine_lr import CosineLRScheduler


def init_lr_scheduler(
        optimizer,
        lr_scheduler="multi_step",  # learning rate scheduler
        stepsize=[20, 40],  # step size to decay learning rate
        gamma=0.1,  # learning rate decay
):
    if lr_scheduler == "single_step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize[0], gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    else:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")


def get_polynomial_decay_schedule_with_warmup(optimizer,
            num_warmup_steps=5, num_training_steps=100, lr_end=1e-7, power=3.0, last_epoch=-1, lr_base_rate=0.2):
    lr_init = optimizer.defaults["lr"] * lr_base_rate
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5,
                                    num_training_steps=100, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_scheduler(optimizer, training_step):
    return CosineAnnealingLR(optimizer, training_step)


def cosine_scheduler_with_warmup(optimizer, max_epoch, base_lr, warmup):
    # num_epochs = cfg.SOLVER.MAX_EPOCHS
    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # type 2
    # lr_min = 0.002 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    # warmup_t = cfg.SOLVER.WARMUP_EPOCHS

    num_epochs = max_epoch
    lr_min = 0.002 * base_lr
    warmup_lr_init = 0.01 * base_lr
    warmup_t = warmup
    noise_range = None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler

