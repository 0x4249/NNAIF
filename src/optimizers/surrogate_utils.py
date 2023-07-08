# Third party imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Local imports
from . import surrogate_fit_loss_functions


@torch.no_grad()
def surrogate_filtered_sampling(surrogate_model,
                                out_transform,
                                x_0,
                                f_0,
                                h,
                                sampler,
                                num_points,
                                iter_max=20):
    """
    Sample points and keep a point if the surrogate model evaluated at the point has a lower value than the
    true function value at the center point (i.e. the point passes the surrogate filter test).
    
    Args:
        surrogate_model (torch.nn.Module): PyTorch surrogate model for filtering sampled points.
        out_transform (callable): transforms surrogate model output to a scalar.
        x_0 (torch.Tensor): center point to sample around.
        f_0 (torch.Tensor): true function value at center point x_0.
        h (float): scale samples by factor of h.
        sampler (callable): argumentless function that generates d-dimensional vectors.
        num_points (int): number of points desired to pass surrogate filter test.
        iter_max (int): maximum number of sampling iterations.
        
    Returns:
        X_accepted (torch.Tensor): tensor containing the accepted x values.
        f_hat_accepted (torch.Tensor): tensor containing the accepted values of the scalar transformed surrogate model output f_hat.
        Y_out_accepted (torch.Tensor): tensor containing the accepted values of the surrogate model output y_out.
    """
    device = x_0.device
    
    # Initialization
    X_accepted = torch.tensor([], device=device)
    Y_out_accepted = torch.tensor([], device=device)
    f_hat_accepted = torch.tensor([], device=device)
    
    point_count = 0
    iter_count = 0   
    
    while point_count < num_points and iter_count < iter_max:
        
        # Choose a new point
        w = sampler().unsqueeze(1).to(device=device)
        x_sample = x_0 + h*w.to(device=device)
        
        # Calculate surrogate model output at sampled point
        y_out = surrogate_model(x_sample)
        f_hat = out_transform(y_out)
        
        # Check that surrogate model output is finite at sampled point
        if f_hat.isfinite():
            
            if f_hat <= f_0:
                X_accepted = torch.cat((X_accepted, x_sample), dim=1)
                Y_out_accepted = torch.cat((Y_out_accepted, y_out.detach()), dim=1)
                f_hat_accepted = torch.cat((f_hat_accepted, f_hat.detach()), dim=0)
                point_count += 1
        
        
        iter_count += 1
    
    
    return X_accepted, f_hat_accepted, Y_out_accepted


def get_surrogate_descent_optimizer(surrogate_descent_opt_dict,
                                    x):
    """
    Create PyTorch optimizer for x according to information in surrogate descent optimizer dictionary 
    (surrogate_descent_opt_dict).
    
    Args:
        surrogate_descent_opt_dict (dict): dictionary storing optimizer information for surrogate descent.
        x (torch.Tensor): parameter to optimize.

    Returns:
        surrogate_descent_opt (torch.optim.Optimizer): PyTorch optimizer for x.
    """
    surrogate_opt_name = surrogate_descent_opt_dict["name"]
    
    
    # PyTorch default optimizers
    # ===============
    if surrogate_opt_name in ["ADAM"]:
        amsgrad = surrogate_descent_opt_dict.get("amsgrad", False)
        lr = surrogate_descent_opt_dict.get("lr", 1e-3)
        betas0 = surrogate_descent_opt_dict.get("betas0", 0.9)
        betas1 = surrogate_descent_opt_dict.get("betas1", 0.999)
        betas = (betas0, betas1)
        surrogate_descent_opt = optim.Adam(params=[x],
                                           amsgrad=amsgrad,
                                           lr=lr,
                                           betas=betas)
        
        
    elif surrogate_opt_name in ["SGD"]:
        lr = surrogate_descent_opt_dict.get("lr", 1e-3)
        surrogate_descent_opt = optim.SGD(params=[x],
                                          lr=lr)
    
    
    else:
        raise ValueError("The surrogate descent optimizer %s is not implemented..." % surrogate_opt_name)
        
        
    return surrogate_descent_opt
    

@torch.enable_grad()
def surrogate_descent_opt_step(surrogate_descent_opt_dict,
                               surrogate_descent_opt, 
                               surrogate_model, 
                               out_transform):
    """
    Define appropriate optimization step procedure for various optimizers when descending surrogate model.
    
    Args:
        surrogate_descent_opt_dict (dict): dictionary storing optimizer information for surrogate descent.
        surrogate_descent_opt (torch.optim.Optimizer): PyTorch optimizer for surrogate descent.
        surrogate_model (torch.nn.Module): PyTorch surrogate model to descend.
        out_transform (callable): transforms surrogate model output to a scalar.
        
    Returns:
        scaled_surrogate_obj (torch.Tensor): new value of scaled surrogate model objective function.
    """
    alpha = surrogate_descent_opt_dict["alpha"]
    surrogate_opt_name = surrogate_descent_opt_dict["name"]
    
    # Zero gradients
    surrogate_descent_opt.zero_grad()
    
    def scaled_surrogate_obj_closure():
        """
        A closure that evaluates the surrogate model objective function scaled by 1/alpha.
        """
        x = surrogate_descent_opt.param_groups[0]['params'][0]
        y_out = surrogate_model(x)
        f_hat = out_transform(y_out)
        scaled_surrogate_obj = f_hat/alpha
        return scaled_surrogate_obj
    
    
    if surrogate_opt_name in ["ADAM", "SGD"]:
        scaled_surrogate_obj = scaled_surrogate_obj_closure()
        scaled_surrogate_obj.backward()
        surrogate_descent_opt.step()
        
        
    else:
        raise ValueError("The surrogate descent optimizer %s is not implemented..." % surrogate_opt_name)
    
    
    return scaled_surrogate_obj
    

def surrogate_descent_tr(surrogate_descent_opt_dict,
                         surrogate_model,
                         out_transform,
                         x_0,
                         h,
                         surrogate_descent_opt=None):
    """
    Minimize the differentiable surrogate model within a trust region.
    
    Args:
        surrogate_descent_opt_dict (dict): dictionary storing optimizer information for surrogate descent.
        surrogate_model (torch.nn.Module): PyTorch surrogate model to minimize.
        out_transform (callable): transforms surrogate model output to a scalar.
        x_0 (torch.Tensor): point at center of trust region.
        h (float): radius of trust region.
        surrogate_descent_opt (torch.optim.Optimizer, optional): PyTorch optimizer for surrogate descent.
        
    Returns:
        X_history (torch.Tensor): tensor containing the values (i.e. iterates) of x.
        f_hat_history (torch.Tensor): tensor containing the values of the scalar transformed surrogate model output f_hat.
        Y_out_history (torch.Tensor): tensor containing the values of the surrogate model output y_out.
    """
    # Initialize x
    device = next(surrogate_model.parameters()).device
    x = nn.Parameter(x_0.clone().detach().to(device=device))
    
    # Create surrogate descent optimizer
    if surrogate_descent_opt is None:
        surrogate_descent_opt = get_surrogate_descent_optimizer(surrogate_descent_opt_dict,
                                                                x)   
    
    iter_count = 0
    iter_max = surrogate_descent_opt_dict["maximum number of surrogate descent iterations"]
    
    while iter_count < iter_max:
        
        if iter_count == 0:          
            with torch.no_grad():
                # Calculate surrogate model output at initial point x_0
                y_out = surrogate_model(x)
                f_hat = out_transform(y_out)
                
                # Record history
                X_history = x_0.clone().detach().to(device=device)
                Y_out_history = y_out.detach()
                f_hat_history = f_hat.detach()
                
                # Check that surrogate model output is finite at initial point
                if torch.sum(y_out.isfinite()) != y_out.ravel().shape[0]:
                    return X_history, f_hat_history, Y_out_history
                
                if not f_hat.isfinite():
                    return X_history, f_hat_history, Y_out_history
        
        
        # Take a step
        surrogate_descent_opt_step(surrogate_descent_opt_dict,
                                   surrogate_descent_opt,
                                   surrogate_model,
                                   out_transform)
        x_proposed = x.detach()
        
        # Check that proposed new point is within trust region
        norm_ord = surrogate_descent_opt_dict["norm order"]
        delta_dist = torch.linalg.norm(x_proposed - x_0, ord=norm_ord)
        
        if delta_dist <= h:
            with torch.no_grad():
                # Calculate surrogate model output at proposed new point
                y_out = surrogate_model(x)
                f_hat = out_transform(y_out)
            
                # Record history
                X_history = torch.cat((X_history, x_proposed), dim=1)
                Y_out_history = torch.cat((Y_out_history, y_out.detach()), dim=1)
                f_hat_history = torch.cat((f_hat_history, f_hat.detach()), dim=0)
            
                # Check that surrogate model output is finite at proposed new point
                if torch.sum(y_out.isfinite()) != y_out.ravel().shape[0]:
                    f_hat_history = torch.cat((f_hat_history, torch.Tensor([torch.nan])), dim=0)
                    return X_history, f_hat_history, Y_out_history
                
                if not f_hat.isfinite():
                    return X_history, f_hat_history, Y_out_history
        
        else:
            return X_history, f_hat_history, Y_out_history


        iter_count += 1
    
    
    return X_history, f_hat_history, Y_out_history
    
    
def get_surrogate_fit_optimizer(surrogate_fit_opt_dict,
                                surrogate_model):
    """
    Create PyTorch optimizer for given PyTorch surrogate model according to information in surrogate fit optimizer dictionary 
    (surrogate_fit_opt_dict).
    
    Args:
        surrogate_fit_opt_dict (dict): dictionary storing optimizer information for fitting surrogate model. 
        surrogate_model (torch.nn.Module): PyTorch surrogate model to fit.
    
    Returns:
        surrogate_fit_opt (torch.optim.Optimizer): PyTorch optimizer for parameters of PyTorch surrogate model to fit. 
    """
    surrogate_fit_opt_name = surrogate_fit_opt_dict["name"]
    
    
    # PyTorch default optimizers
    # ===============
    if surrogate_fit_opt_name in ["ADAM"]:
        amsgrad = surrogate_fit_opt_dict.get("amsgrad", False)
        lr = surrogate_fit_opt_dict.get("lr", 1e-3)
        betas0 = surrogate_fit_opt_dict.get("betas0", 0.9)
        betas1 = surrogate_fit_opt_dict.get("betas1", 0.999)
        betas = (betas0, betas1)        
        surrogate_fit_opt = optim.Adam(params=surrogate_model.parameters(),
                                       amsgrad=amsgrad,
                                       lr=lr,
                                       betas=betas)
        
        
    elif surrogate_fit_opt_name in ["SGD"]:
        lr = surrogate_fit_opt_dict.get("lr", 1e-3)
        surrogate_fit_opt = optim.SGD(params=surrogate_model.parameters(),
                                      lr=lr)
    
    
    else:
        raise ValueError("The surrogate fit optimizer %s is not implemented..." % surrogate_fit_opt_name)
        
        
    return surrogate_fit_opt
    

def get_param_regularizer_closure(param_regularizer_dict,
                                  surrogate_model):
    """
    Setup surrogate model parameter regularization function closure.
    
    Args:
        param_regularizer_dict (dict): dictionary containing parameter regularization function type and settings.
        surrogate_model (torch.nn.Module): PyTorch surrogate model to regularize.
    
    Returns:
        param_regularizer_closure (callable): a closure that evaluates the surrogate model parameter regularization function.
    """
    param_reg_type = param_regularizer_dict["type"]
    
    if param_reg_type in ["model specific", "Model Specific"]:
        alpha_x = param_regularizer_dict["alpha_x"]
        kwargs = param_regularizer_dict["kwargs"]
        
        def param_regularizer_closure():
            reg = surrogate_model.regularization(**kwargs)
            reg = alpha_x*reg
            return reg
    
    
    elif param_reg_type in ["None", "none"]:
    
        device = next(surrogate_model.parameters()).device
            
        def param_regularizer_closure():
            return torch.zeros(1, device=device).squeeze()
    
    
    else:
        raise ValueError("The parameter regularization type %s is not implemented..." % param_reg_type)
    
    
    return param_regularizer_closure
    
    
def get_model_output_regularizer(model_output_regularizer_dict,
                                 surrogate_model):
    """
    Setup surrogate model output regularization function.
    
    Args:
        model_output_regularizer_dict (dict): dictionary containing model output regularization function type and 
                                              settings.
        surrogate_model (torch.nn.Module): PyTorch surrogate model to regularize the output of.
        
    Returns:
        model_output_regularization_function (callable): model output regularization function.
    """
    model_output_reg_type = model_output_regularizer_dict["type"]
    
    if model_output_reg_type in ["None", "none"]:
    
        device = next(surrogate_model.parameters()).device
        
        def model_output_regularization_function(output):
            return torch.zeros(1, device=device).squeeze()
    
    
    else:
        raise ValueError("The model output regularization type %s is not implemented..." % model_output_reg_type)


    return model_output_regularization_function
    

@torch.no_grad()
def compute_scaled_surrogate_fit_obj_on_dataset(surrogate_fit_opt_dict,
                                                surrogate_model,
                                                out_transform,
                                                dataset,
                                                batch_size,
                                                verbose=False):
    """
    Compute specified scaled surrogate fit objective function over specified dataset.
    
    Args:
        surrogate_fit_opt_dict (dict): dictionary storing optimizer information for fitting surrogate model.
        surrogate_model (torch.nn.Module): PyTorch surrogate model.
        out_transform (callable): transforms surrogate model output to a scalar.
        dataset (torch.utils.data.Dataset): dataset from which to load the data.
        batch_size (int): batch size. 
        verbose (bool): print additional information, useful for debugging (default: False).
        
    Returns:
        scaled_surrogate_fit_obj_function_mean (torch.Tensor): mean of scaled surrogate fit objective function computed
                                                               over specified dataset.
        scaled_surrogate_fit_loss_function_mean (torch.Tensor): mean of scaled surrogate fit loss function computed
                                                                over specified dataset.
    """
    alpha = surrogate_fit_opt_dict["alpha"]
    device = next(surrogate_model.parameters()).device
    
    # Surrogate fit loss function setup
    # ===============
    surrogate_fit_loss_func_dict = surrogate_fit_opt_dict["loss function dictionary]
    surrogate_fit_loss_function = surrogate_fit_loss_functions.get_surrogate_fit_loss_function(surrogate_fit_loss_func_dict)
    
    # Parameter regularizer setup
    # ===============
    param_regularizer_dict = surrogate_fit_opt_dict["parameter regularization dictionary"]
    param_regularizer_closure = get_param_regularizer_closure(param_regularizer_dict,
                                                              surrogate_model)
    
    # Surrogate model output regularizer setup
    # ===============
    surrogate_model_output_regularizer_dict = surrogate_fit_opt_dict["model output regularizer dictionary"]
    surrogate_model_output_regularization_function = get_model_output_regularizer(surrogate_model_output_regularizer_dict,
                                                                                  surrogate_model)
    
    
    surrogate_model.eval()
    
    loader = data.DataLoader(dataset,
                             drop_last=False,
                             batch_size=batch_size)
    
    scaled_surrogate_fit_obj_function_sum = torch.zeros(1, device=device)
    scaled_surrogate_fit_loss_function_sum = torch.zeros(1, device=device)
    
    for batch in loader:
        X = batch[0].T.to(device)
        fs = batch[1].to(device)
        this_batch_size = fs.shape[0]
        
        predictions = out_transform(surrogate_model(X)) # need to broadcast out_transform here 
        scaled_surrogate_fit_loss = surrogate_fit_loss_function(predictions, fs)/alpha
        scaled_surrogate_param_regularizer = param_regularizer_closure()/alpha
        scaled_surrogate_model_output_regularizer = surrogate_model_output_regularization_function(predictions)/alpha
        
        scaled_surrogate_fit_obj_fun = 0
        scaled_surrogate_fit_obj_fun += scaled_surrogate_fit_loss
        scaled_surrogate_fit_obj_fun += scaled_surrogate_param_regularizer
        scaled_surrogate_fit_obj_fun += scaled_surrogate_model_output_regularizer
        
        scaled_surrogate_fit_obj_function_sum += this_batch_size*scaled_surrogate_fit_obj_fun
        scaled_surrogate_fit_loss_function_sum += this_batch_size*scaled_surrogate_fit_loss
    
    
    scaled_surrogate_fit_obj_function_mean = scaled_surrogate_fit_obj_function_sum/len(loader.dataset)
    scaled_surrogate_fit_loss_function_mean = scaled_surrogate_fit_loss_function_sum/len(loader.dataset)
    
    if verbose:
        print(scaled_surrogate_fit_obj_function_mean, scaled_surrogate_fit_loss_function_mean)
    
    return scaled_surrogate_fit_obj_function_mean, scaled_surrogate_fit_loss_function_mean


@torch.enable_grad()
def surrogate_fit_opt_step(surrogate_fit_opt_dict,
                           surrogate_fit_opt, 
                           surrogate_model, 
                           out_transform, 
                           batch,
                           loss_function,
                           param_regularizer_closure,
                           surrogate_model_output_regularization_function):
    """
    Define appropriate optimization step procedure for various optimizers when fitting surrogate model.
    
    Args:
        surrogate_fit_opt_dict (dict): dictionary storing optimizer information for fitting surrogate model. 
        surrogate_fit_opt (torch.optim.Optimizer): PyTorch optimizer for fitting surrogate model.
        surrogate_model (torch.nn.Module): PyTorch surrogate model to fit.
        out_transform (callable): transforms surrogate model output to a scalar.
        batch (list): batch as list of examples.
        loss_function (callable): loss function.
        param_regularizer_closure (callable): a closure that evaluates the surrogate model parameter regularization function.
        surrogate_model_output_regularization_function (callable): regularization function for surrogate model output.
    
    Returns:
        scaled_surrogate_fit_obj (torch.Tensor): scaled surrogate model fit objective function.
    """
    alpha = surrogate_fit_opt_dict["alpha"]
    surrogate_fit_opt_name = surrogate_fit_opt_dict["name"]
    device = next(surrogate_model.parameters()).device
    
    X = batch[0].T.to(device)
    fs = batch[1].to(device)
    
    # Zero gradients
    surrogate_fit_opt.zero_grad()
    
    def transformed_model_closure():
        """
        A closure that evaluates the surrogate model output for the current batch transformed by out_transform.
        """
        return out_transform(surrogate_model(X)) # need to broadcast out_transform here 
    
    
    def scaled_surrogate_fit_batch_loss_function(predictions):
        """
        Surrogate fit loss with labels from current batch fixed and scaled by 1/alpha.
        """
        return loss_function(predictions, fs)/alpha
        
        
    def scaled_param_regularizer_closure():
        """
        Surrogate model parameter regularization function closure scaled by 1/alpha.
        """
        return param_regularizer_closure()/alpha
    
    
    def scaled_model_output_regularization_function(output):
        """
        Surrogate model output regularization function scaled by 1/alpha.
        """
        return surrogate_model_output_regularization_function(output)/alpha
    
    
    def scaled_surrogate_fit_obj_closure():
        """
        A closure that evaluates the surrogate fit objective function scaled by 1/alpha.
        """
        transformed_model_output = transformed_model_closure()
        scaled_loss_fun = scaled_surrogate_fit_batch_loss_function(transformed_model_output)
        scaled_param_regularizer = scaled_param_regularizer_closure()
        scaled_model_output_regularizer = scaled_model_output_regularization_function(transformed_model_output)
        
        scaled_surrogate_fit_obj_fun = scaled_loss_fun + scaled_param_regularizer + scaled_model_output_regularizer
        
        return scaled_surrogate_fit_obj_fun
    
    
    if surrogate_fit_opt_name in ["ADAM", "SGD"]:
        scaled_surrogate_fit_obj = scaled_surrogate_fit_obj_closure()
        scaled_surrogate_fit_obj.backward()
        surrogate_fit_opt.step()
        
        
    else:
        raise ValueError("The surrogate fit optimizer %s is not implemented..." % surrogate_fit_opt_name)
    
    
    return scaled_surrogate_fit_obj
    
    
def fit_surrogate_model(surrogate_fit_opt_dict,
                        surrogate_model,
                        out_transform,
                        fs,
                        X,
                        surrogate_fit_opt=None):
    """
    Fit the surrogate model.
    
    Args:
        surrogate_fit_opt_dict (dict): dictionary storing optimizer information for fitting surrogate model. 
        surrogate_model (torch.nn.Module): PyTorch surrogate model to fit.
        out_transform (callable): transforms surrogate model output to a scalar.
        fs (torch.Tensor): tensor containing true function f values.
        X (torch.Tensor): two dimensional tensor containing x values as its columns.
        surrogate_fit_opt (torch.optim.Optimizer, optional): PyTorch optimizer for fitting surrogate model.
        
    Returns:
        scaled_surrogate_fit_obj_means (list): means of scaled surrogate fit objective function computed over entire 
                                               dataset given by X and fs.
        scaled_surrogate_fit_loss_means (list): means of scaled surrogate fit loss function computed over entire 
                                                dataset given by X and fs.
    """    
    surrogate_fit_opt_name = surrogate_fit_opt_dict["name"]
    
    # Dataset setup
    # ===============]
    
    dataset = data.TensorDataset(X.T, fs)
    train_loader = data.DataLoader(dataset,
                                   drop_last=False,
                                   shuffle=True,
                                   batch_size=surrogate_fit_opt_dict["batch size"])
    
    # Surrogate fit loss function setup
    # ===============
    surrogate_fit_loss_func_dict = surrogate_fit_opt_dict["loss function dictionary"]
    surrogate_fit_loss_function = surrogate_fit_loss_functions.get_surrogate_fit_loss_function(surrogate_fit_loss_func_dict)
    
    # Parameter regularizer setup
    # ===============
    param_regularizer_dict = surrogate_fit_opt_dict["parameter regularization dictionary"]
    param_regularizer_closure = get_param_regularizer_closure(param_regularizer_dict,
                                                              surrogate_model)
    
    # Surrogate model output regularizer setup
    # ===============
    surrogate_model_output_regularizer_dict = surrogate_fit_opt_dict["model output regularizer dictionary"]
    surrogate_model_output_regularization_function = get_model_output_regularizer(surrogate_model_output_regularizer_dict,
                                                                                  surrogate_model)
    
    # Surrogate fit optimizer setup
    # ===============
    if surrogate_fit_opt is None:
        surrogate_fit_opt = get_surrogate_fit_optimizer(surrogate_fit_opt_dict,
                                                        surrogate_model)
    
    scaled_surrogate_fit_obj_means = []
    scaled_surrogate_fit_loss_means = []
    
    max_epochs = surrogate_fit_opt_dict["max epochs"]
    loss_tol = surrogate_fit_opt_dict["loss tolerance"]
    
    # Main data fitting loop
    # ===============
    for e in range(max_epochs):
        
        for batch in train_loader:
            this_batch_size = batch[1].shape[0]
            
            # Take a step
            surrogate_fit_obj = surrogate_fit_opt_step(surrogate_fit_opt_dict,
                                                       surrogate_fit_opt,
                                                       surrogate_model,
                                                       out_transform,
                                                       batch,
                                                       surrogate_fit_loss_function,
                                                       param_regularizer_closure,
                                                       surrogate_model_output_regularization_function)
            
        # Compute average surrogate fit loss over entire dataset
        out = compute_scaled_surrogate_fit_obj_on_dataset(surrogate_fit_opt_dict,
                                                          surrogate_model,
                                                          out_transform,
                                                          dataset,
                                                          batch_size=surrogate_fit_opt_dict["batch size"])
        scaled_surrogate_fit_obj_function_mean = out[0]
        scaled_surrogate_fit_loss_function_mean = out[1]
        
        # End training early if values become NaN or Inf
        if not scaled_surrogate_fit_obj_function_mean.isfinite():
            break
        
        # Record results over entire dataset
        scaled_surrogate_fit_obj_means.append(scaled_surrogate_fit_obj_function_mean.item())
        scaled_surrogate_fit_loss_means.append(scaled_surrogate_fit_loss_function_mean.item())
            
        # End training early if average surrogate fit loss below tolerance threshold
        if scaled_surrogate_fit_loss_function_mean < loss_tol:
            break
            
    
    return scaled_surrogate_fit_obj_means, scaled_surrogate_fit_loss_means
    
    
