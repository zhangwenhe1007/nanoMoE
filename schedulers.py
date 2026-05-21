import math

def get_warmup_cosine_lr(step, base_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + decay * (base_lr - min_lr)


class WarmupCosineScheduler:
    def __init__(self, optimizer, base_lr, min_lr, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step):
        return get_warmup_cosine_lr(
            step,
            self.base_lr,
            self.min_lr,
            self.warmup_steps,
            self.max_steps,
        )

    def step(self, step):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr