from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def lr_scheduler(optimizer, step_size, gamma):
    """Create LR scheduler.

    Args:
        optimizer: Model optimizer.
        step_size: Frequency for changing learning rate.
        gamma: Factor for changing learning rate.
    
    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma)

def reduceLRonplateau(optimizer, factor=0.1, patience=10, verbose=False, min_lr=0):

    return ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr
    )
