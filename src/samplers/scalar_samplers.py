# Third party imports
import torch 


def zero_sampler():
    """
    Function returning a function sampling scalar zero noise.
    
    Returns:
        sampler (callable): Argumentless function returning the scalar value zero.
    """
    def sampler():
        return 0.0


    return sampler
    
    
def unit_sampler():
    """
    Function returning a function sampling scalar unit noise.
    
    Returns:
        sampler (callable): Argumentless function returning the scalar value one.
    """
    def sampler():
        return 1.0


    return sampler 
    
    
def symmetric_uniform_random_sampler(epsilon=1e-2):
    """
    Function returning a function sampling from the scalar symmetric zero mean uniform distribution.
    
    Args:
        epsilon (float): half of width of uniform distribution.
        
    Returns:
        sampler (callable): Argumentless function returning a sample from specified scalar 
                            symmetric zero mean uniform distribution.
    """
    def sampler():
        return -epsilon + 2*epsilon*torch.rand(1)


    return sampler
    

def gaussian_random_sampler(mu=0.0, 
                            sigma=1.0):
    """
    Function returning a function sampling from a scalar Gaussian distribution.
    
    Args:
        mu (float): mean of Gaussian distribution.
        sigma (float): standard deviation of Gaussian distribution.
        
    Returns:
        sampler (callable): Argumentless function returning a sample from specified scalar Gaussian distribution.
    """
    def sampler():
        return mu + sigma*torch.randn(1)


    return sampler    
    
    
