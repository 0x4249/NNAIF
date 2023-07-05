# Third party imports
import pycutest
import torch

# Local imports
from . import cutest


def get_cutest_problem(problem_name, 
                       sif_params,
                       dtype=torch.float):
    """
    Retrieve a CUTEst problem as a PyTorch module.
    
    Args:
        problem_name (string): name of CUTEst problem.
        sif_params (dict): dictionary containing SIF parameters to use with the CUTEst problem.
        dtype (dtype or string): the desired data type (default: torch.float).
        
    Returns:
        cutest_model (torch.nn.Module): CUTEst problem as a PyTorch module.
    """
    problem = pycutest.import_problem(problem_name, 
                                      sifParams=sif_params)
    cutest_model = cutest.CUTEstProblem(problem,
                                        dtype=dtype)
    
    return cutest_model
    
    
def get_cutest_problem_additive_noise(problem_name, 
                                      sif_params,
                                      obj_noise_func,
                                      grad_noise_func,
                                      dtype=torch.float):
    """
    Retrieve CUTEst problem with additive noise as a PyTorch module.
    
    Args:
        problem_name (string): name of CUTEst problem.
        sif_params (dict): dictionary containing SIF parameters to use with the CUTEst problem.
        obj_noise_func (callable): argumentless function drawing scalar samples to add to CUTEst objective function.
        grad_noise_func (callable): argumentless function drawing vector samples to add to CUTEst gradient. 
        dtype (dtype or string): the desired data type (default: torch.float).
        
    Returns:
        cutest_model (torch.nn.Module): CUTEst problem with additive noise as a PyTorch module.
    """                                          
    problem = pycutest.import_problem(problem_name, 
                                      sifParams=sif_params)
    cutest_model = cutest.CUTEstProblemAdditiveNoise(problem, 
                                                     obj_noise_func, 
                                                     grad_noise_func,
                                                     dtype=dtype)
    
    return cutest_model        
    

