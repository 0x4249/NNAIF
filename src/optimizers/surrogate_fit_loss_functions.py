# Third party imports
import torch


def get_surrogate_fit_loss_function(surrogate_fit_loss_func_dict):
    """
    Get surrogate fit loss function according to information in surrogate fit loss function dictionary 
    (surrogate_fit_loss_func_dict).
    
    Args:
        surrogate_fit_loss_func_dict (dict): dictionary containing surrogate fit loss function name and settings.
        
    Returns:
        surrogate_fit_loss_function (callable): surrogate fit loss function.
    """
    surrogate_fit_loss_func_name = surrogate_fit_loss_func_dict["name"]
    
    if surrogate_fit_loss_func_name in ["squared error"]:
        surrogate_fit_loss_function = squared_error
        
        
    else:
        raise ValueError("The surrogate fit loss function %s is not implemented..." % surrogate_fit_loss_func_name)
        
        
    return surrogate_fit_loss_function
    
    
def squared_error(predictions,
                  labels,
                  reduction="mean"):
    """
    Compute the squared error.
    
    Args:
        predictions (torch.Tensor): predictions.
        labels (torch.Tensor): true labels.
        reduction (string): 'none', 'mean', or 'sum' (default: "mean").
        
    Returns:
        squared_error (torch.Tensor): squared error.
    """
    criterion = torch.nn.MSELoss(reduction=reduction)
    squared_error = criterion(predictions.view(-1), labels.view(-1))
    
    return squared_error


