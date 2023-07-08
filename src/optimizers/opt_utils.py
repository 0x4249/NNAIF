# Standard library imports
import contextlib

# Third party imports
import numpy as np
import torch
import torch.cuda


def is_legal(v):
    """
    Check that a PyTorch tensor does not contain NaN or Inf values.
    
    Args:
        v (torch.Tensor): PyTorch tensor to check.
        
    Returns:
        legal (bool): False if the tensor contains any NaN or Inf values, True otherwise.
    """
    legal = not torch.isnan(v).any() and not torch.isinf(v).any()
    return legal
    

def check_armijo_condition(step_size, 
                           loss, 
                           loss_try,
                           flat_grad, 
                           flat_update_direction, 
                           c=1e-4, 
                           tau=0.1,
                           epsilon_a=0.0):
    """
    Check Armijo (i.e. sufficient decrease) condition, backtracking if the condition is not satisfied.
    
    Args:
        step_size (float): step size along flat update direction. 
        loss (float): loss at original/starting point.
        loss_try (float): loss at new point to try (i.e. proposed point).
        flat_grad (torch.Tensor): parameter gradients flattened into one dimensional tensor. 
        flat_update_direction (torch.Tensor): parameter update direction flattened into one dimensional tensor. 
        c (float): decrease factor 0 < c < 1 (default: 1e-4).
        tau (float): backtracking factor 0 < tau < 1 (default: 0.1).
        epsilon_a (float): relax the Armijo condition by a factor of 2*epsilon_a (default: 0.0).
        
    Returns:
        found (bool): True if the Armijo condition is satisfied, False otherwise.
        step_size (float): step size after applying backtracking (backtracking only occurs if found is False).
    """
    found = False
    
    flat_grad_dot_flat_update_direction = flat_grad.dot(flat_update_direction)
    break_condition = loss_try - (loss + step_size*c*flat_grad_dot_flat_update_direction + 2*epsilon_a)
    
    if (break_condition <= 0):
        found = True
        
    else:
        # Decrease step size via backtracking
        step_size = step_size*tau
        
    return found, step_size 
    

@contextlib.contextmanager
def random_seed_torch(seed, 
                      device=0):
    """
    Context manager for setting the seed of both CPU and specified GPU device random number generators. Exiting the 
    context restores the CPU and specified GPU device random number generators back to their initial states. 
    
    Args:
        seed (int): the desired seed.
        device (torch.device or int): the specified GPU device (default: 0).
    """
    # Store initial CPU random number generator state
    cpu_rng_state = torch.get_rng_state()
    
    # Store initial specified GPU device random number generator state
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(device=device)
    
    # Seed CPU 
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Seed all GPU devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    try:
        yield
    finally:
        # Set CPU random number generator back to its initial state
        torch.set_rng_state(cpu_rng_state)
        
        # Set specified GPU device random number generator back to its initial state
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, 
                                     device)
                                     

