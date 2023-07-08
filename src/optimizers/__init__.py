# Local imports
from . import cmaes
from . import emna
from . import imfil
from . import nnaif
from . import sgpgd
from . import surrogate_models

# Third party imports
import torch


def get_optimizer(opt_dict, 
                  model):
    """
    Create PyTorch optimizer for given PyTorch model according to information in optimizer dictionary (opt_dict).
    
    Args:
        opt_dict (dict): dictionary containing optimizer name and settings. 
        model (torch.nn.Module): PyTorch model to optimize. 
        
    Returns:
        opt (torch.optim.Optimizer): PyTorch optimizer for parameters of PyTorch model to optimize. 
    """
    opt_name = opt_dict["name"]
    
    
    # Custom optimizers
    # ===============
    if opt_name in ["CMA-ES"]:
        population_size = opt_dict.get("population size", 20)
        
        if population_size in ["d"]:
            population_size = sum(p.numel() for p in model.parameters())
            
        elif population_size in ["d+1"]:
            population_size = sum(p.numel() for p in model.parameters()) + 1
            
        sigma_0 = opt_dict.get("sigma_0", 1)
        active = opt_dict.get("active", True)
        verbose = opt_dict.get("verbose", False)
        opt = cmaes.CMAES(params=model.parameters(),
                          population_size=population_size,
                          sigma_0=sigma_0,
                          active=active,
                          verbose=verbose)
        

    elif opt_name in ["EMNA"]:
        sigma_0 = opt_dict.get("sigma_0", 1.0)
        gamma = opt_dict.get("gamma", 1.0)
        verbose = opt_dict.get("verbose", False)
        opt = emna.EMNA(params=model.parameters(),
                        sigma_0=sigma_0,
                        gamma=gamma,
                        verbose=verbose)
        
    
    elif opt_name in ["IMFIL"]:
        H_0 = torch.tensor([])
        
        h_0 = opt_dict.get("h_0", 1.0)
        hessian_approx = opt_dict.get("hessian approximation", "BFGS")
        
        if hessian_approx in ["BFGS"]:
            initial_inv_hessian = opt_dict["initial inverse hessian"]
            
            if initial_inv_hessian in ["I"]:
                d = sum(p.numel() for p in model.parameters())
                H_0 = torch.eye(d)
        
        verbose = opt_dict.get("verbose", False)
        opt = imfil.IMFIL(params=model.parameters(),
                          h_0=h_0,
                          hessian_approx=hessian_approx,
                          H_0=H_0,
                          verbose=verbose)
    
    
    elif opt_name in ["NNAIF"]:
        numel = sum(p.numel() for p in model.parameters())
        surrogate_model_dict = opt_dict["surrogate model dictionary"]
        surrogate_model_type = surrogate_model_dict["type"]
        
        if surrogate_model_type in ["RESNET EULER"]:
            
            d_layer = surrogate_model_dict["d_layer"]      
            d_out = surrogate_model_dict["d_out"]
            num_square_layers = surrogate_model_dict["number of square layers"]
            dt = surrogate_model_dict["dt"]
            sigma = surrogate_model_dict["sigma"]
            
            surrogate_model = surrogate_models.RESNETEULER(d_in=numel,
                                                           d_layer=d_layer,
                                                           d_out=d_out,
                                                           num_square_layers=num_square_layers,
                                                           dt=dt,
                                                           sigma=sigma)
        
        
        else:
            raise ValueError("The surrogate model %s is not implemented for NNAIF..." % surrogate_model_type)
        
        
        H_0 = torch.tensor([])
        
        surrogate_fit_opt_dict = opt_dict["surrogate fit optimizer dictionary"]
        h_0 = opt_dict.get("h_0", 1.0)
        hessian_approx = opt_dict.get("hessian approximation", "BFGS")
        
        if hessian_approx in ["BFGS"]:
            initial_inv_hessian = opt_dict["initial inverse hessian"]
            
            if initial_inv_hessian in ["I"]:
                d = sum(p.numel() for p in model.parameters())
                H_0 = torch.eye(d)
        
        verbose = opt_dict.get("verbose", False)
        opt = nnaif.NNAIF(params=model.parameters(),
                          surrogate_model=surrogate_model,
                          surrogate_fit_opt_dict=surrogate_fit_opt_dict,
                          h_0=h_0,
                          hessian_approx=hessian_approx,
                          H_0=H_0,
                          verbose=verbose)


    elif opt_name in ["SG-PGD"]:
        line_search = opt_dict.get("line search", "Armijo")
        
        H_0 = torch.tensor([])
        
        hessian_approx = opt_dict.get("hessian approximation", "BFGS")
        
        if hessian_approx in ["BFGS"]:
            initial_inv_hessian = opt_dict["initial inverse hessian"]
            
            if initial_inv_hessian in ["I"]:
                d = sum(p.numel() for p in model.parameters())
                H_0 = torch.eye(d)
        
        verbose = opt_dict.get("verbose", False)
        opt = sgpgd.SGPGD(params=model.parameters(),
                         line_search=line_search,
                         hessian_approx=hessian_approx,
                         H_0=H_0,
                         verbose=verbose)
    
    
    else:
        raise ValueError("The optimizer %s is not implemented..." % opt_name)
        
        
    return opt
    
    
