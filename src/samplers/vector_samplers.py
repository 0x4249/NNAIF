# Third party imports
import torch 


def multivariate_normal_random_sampler(mu, 
                                       Sigma):
    """
    Function returning a function sampling from a multivariate normal 
    distribution with mean mu and covariance Sigma.

    Args:
        mu (torch.Tensor): mean of the multivariate normal distribution.
        Sigma (torch.Tensor): covariance matrix of the multivariate normal distribution.
        
    Returns:
        sampler (callable): argumentless function returning a sample from specified multivariate normal distribution.
    """
    d = mu.shape[0]
    L = torch.linalg.cholesky(Sigma)
    
    def multivariate_normal_random_sample():
        y = torch.randn(d)
        Y = L @ y + mu
        return Y
        
        
    def sampler():
        return multivariate_normal_random_sample()


    return sampler


def spherical_ball_uniform_random_sampler(d=2, 
                                          r=1.0):
    """
    Function returning a function sampling the uniform distribution
    inside the d-dimensional spherical (i.e. 2-norm) ball with radius r centered at zero.
    
    Args:
        d (int): dimension of the ball.
        r (float): radius of the ball.
        
    Returns:
        sampler (callable): argumentless function returning a sample from specified uniform distribution inside a
                            spherical ball centered at zero.
    """
    def spherical_ball_uniform_random_sample(d,
                                             r):
        y = torch.randn(d)
        Y = y/torch.linalg.norm(y)
        U = torch.pow(torch.rand(1),1/d)
        return r*U*Y


    def sampler():
        return spherical_ball_uniform_random_sample(d,
                                                    r)


    return sampler


def spherical_surface_uniform_random_sampler(d=2, 
                                             r=1.0):
    """
    Function returning a function sampling the uniform distribution
    on the surface of the d-dimensional spherical (i.e. 2-norm) ball with
    radius r centered at zero.
    
    Args:
        d (int): dimension of the ball.
        r (float): radius of the ball.
        
    Returns:
        sampler (callable): argumentless function returning a sample from specified uniform distribution on the surface 
                            of a spherical ball centered at zero.
    """
    def spherical_surface_uniform_random_sample(d,
                                                r):
        y = torch.randn(d)
        Y = y/torch.linalg.norm(y)
        return r*Y


    def sampler():
        return spherical_surface_uniform_random_sample(d,
                                                       r)


    return sampler
    

