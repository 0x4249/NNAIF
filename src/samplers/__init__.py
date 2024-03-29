# Local imports
from . import scalar_samplers
from . import vector_samplers

# Third party imports
import torch


def get_sampler(sampler_dict):
    """
    Create sampler function according to information in sampler dictionary (sampler_dict).

    Args:
        sampler_dict (dict): dictionary containing sampler name and parameters.
        
    Returns:
        sampler (callable): function that generates samples of specified type.
    """
    sampler_name = sampler_dict["name"]
    
    
    if sampler_name in ["zero sampler"]:
        sampler = scalar_samplers.zero_sampler()
        
        
    elif sampler_name in ["unit sampler"]:
        sampler = scalar_samplers.unit_sampler()
    
    
    elif sampler_name in ["symmetric uniform random sampler"]:
        r = sampler_dict["radius"]
        sampler = scalar_samplers.symmetric_uniform_random_sampler(epsilon=r)
        
        
    elif sampler_name in ["gaussian random sampler"]:
        mu = sampler_dict["mu"]
        sigma = sampler_dict["sigma"]
        sampler = scalar_samplers.gaussian_random_sampler(mu=mu,
                                                          sigma=sigma)
    
    
    elif sampler_name in ["spherical ball uniform random sampler"]:
        d = sampler_dict["dimension"]
        r = sampler_dict["radius"]
        sampler = vector_samplers.spherical_ball_uniform_random_sampler(d=d,
                                                                        r=r)
        
        
    elif sampler_name in ["spherical surface uniform random sampler"]:
        d = sampler_dict["dimension"]
        r = sampler_dict["radius"]
        sampler = vector_samplers.spherical_surface_uniform_random_sampler(d=d,
                                                                           r=r)
    
    
    return sampler
    

def get_vector_sampler(sampler_dict):
    """
    Create vector sampler function according to information in sampler dictionary (sampler_dict).
    
    Args:
        sampler_dict (dict): dictionary containing sampler name and parameters.
        
    Returns:
        sampler (callable): function that generates samples of specified type.
    """
    sampler_name = sampler_dict["name"]
    
    
    if sampler_name in ["spherical ball uniform random sampler"]:
        dim = sampler_dict["dimension"]
        sampler = vector_samplers.spherical_ball_uniform_random_sampler(d=dim)
        
        
    elif sampler_name in ["spherical surface uniform random sampler"]:
        dim = sampler_dict["dimension"]
        sampler = vector_samplers.spherical_surface_uniform_random_sampler(d=dim)   
    
    
    elif sampler_name in ["standard multivariate normal random sampler"]:
        sampler = vector_samplers.multivariate_normal_random_sampler(mu=torch.zeros(dim),
                                                                     Sigma=torch.eye(dim))
    
    
    return sampler    
    
    
