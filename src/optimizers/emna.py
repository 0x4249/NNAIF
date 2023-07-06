# Standard library imports
import time
from functools import reduce

# Third party imports
import torch
import torch.optim as optim

# Local imports
from . import opt_utils


class EMNA(optim.Optimizer):
    """
    Implements the Estimation of Multivariate Normal Algorithm (EMNA) optimization algorithm as a PyTorch optimizer.
    For a description of EMNA, see https://en.wikipedia.org/wiki/Cross-entropy_method#Continuous_optimization—example.
    
    .. warning::
        This optimizer doesn't support per-parameter options and parameter groups (there can only be one group).
        
    .. warning::
        Right now all parameters have to be on a single device. 
        
    Args:
        params (list): list of torch.Tensor objects containing parameters to optimize. 
        sigma_0 (float): initialize the covariance matrix estimate as (sigma_0^2 * identity matrix). 
        gamma (float): regularize the covariance matrix estimate by adding (gamma * identity matrix). This ensures 
                       the regularized covariance matrix is positive definite when gamma > 0 (default: 1.0).
        dtype (dtype or string): the desired data type (default: torch.float).
        verbose (bool): print additional information, useful for debugging (default: False).
    """
    def __init__(self,
                 params,
                 sigma_0,
                 gamma=1.0,
                 dtype=torch.float,
                 verbose=False):

        # Check that covariance matrix regularization parameter gamma is valid
        if not gamma >= 0.0:
            raise ValueError("Invalid covariance matrix regularization parameter gamma (must be >= 0): {}".format(gamma))
        
        defaults = dict(sigma_0=sigma_0,
                        gamma=gamma,
                        dtype=dtype,
                        verbose=verbose)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("EMNA doesn't support per-parameter options (parameter groups)")
            
        self._params = self.param_groups[0]['params']
        self._device = self._params[0].device
        self._numel_cache = None
        
        # NOTE: EMNA has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        d = self._calc_numel()
        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0) # iteration count
        state.setdefault('prev_obj_min', None) # previous objective function minimum value
        state.setdefault('obj_closure_eval_count', 0) # count of objective function closure evaluations at current iteration
        state.setdefault('Sigma', ((sigma_0**2) + gamma)*torch.eye(d, device=self._device)) # sampling distribution covariance

        
    def _calc_numel(self):
        """
        Calculate the number of elements to be optimized and cache result. 
        
        Returns:
            _numel_cache (): total number of elements to be optimized.
        """
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache      


    def _gather_flat_params(self):
        """
        Concatenate (i.e. flatten) parameters into one dimensional tensor.
        
        Returns:
            flat_params (torch.Tensor): one dimensional tensor containing parameters.
        """
        views = []
        for p in self._params:
            if p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
            
        flat_params = torch.cat(views, 0)
        return flat_params
                             

    def _update_params_to_point(self, 
                                point):
        """
        Update parameters being optimized to given point.
        
        Args:
            point (torch.Tensor): one dimensional tensor containing point to update to.
        """
        offset = 0
        for p in self._params:
            numel = p.numel()
            # use view to avoid deprecated pointwise semantics
            p.copy_(point[offset:offset + numel].view_as(p))
            offset += numel
            
        assert offset == self._calc_numel() 

        
    def _clone_param(self):
        """
        Clone parameters being optimized.
        
        Returns:
            cloned_parameters (list): list of torch.Tensor objects containing cloned parameters.
        """
        cloned_parameters = [p.clone(memory_format=torch.contiguous_format) for p in self._params]
        return cloned_parameters


    def _set_param(self, 
                   params_data):
        """
        Set parameters being optimized to given values.
        
        Args:
            params_data (list): list of torch.Tensor objects containing values to set parameters to.
        """
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata) 
                        
    
    def _obj_point_evaluate(self, 
                            obj_closure, 
                            flat_point):
        """
        Evaluate the objective function at a given point.
        
        Args:
            obj_closure (callable): A closure that evaluates the objective function.
            flat_point (torch.Tensor): one dimensional tensor containing point to evaluate at. 
            
        Returns:
            obj (float): Value of objective function at given point.
        """
        state = self.state[self._params[0]]
        
        # store initial point in parameter space
        x = self._clone_param() 
        
        # set to new point in parameter space
        self._update_params_to_point(flat_point)
        
        with torch.no_grad():
            # evaluate objective function only
            obj = obj_closure() 
            state['obj_closure_eval_count'] += 1
            
            # restore parameters to initial values
            self._set_param(x) 
                
            obj = float(obj)                
                
            return obj
                                                               
        
    @torch.no_grad()
    def step(self,
             obj_closure,
             num_new_samples=100,
             num_kept_samples=10):
        """
        Performs a single EMNA optimization step.
        
        Args:
            obj_closure (callable): A closure that evaluates the objective function.
            num_new_samples (int): total number of samples to draw from multivariate normal distribution (default: 100).
            num_kept_samples (int): number of elite samples to keep (default: 10).
            
        Returns:
            obj_min (float): smallest measured value of objective function.
        """
        # Check that number of new samples is valid
        if not num_new_samples > 0:
            raise ValueError("Invalid number of new samples: {}".format(num_new_samples)) 
            
        # Check that number of kept samples is valid
        if not num_kept_samples > 0 or not num_kept_samples <= num_new_samples:
            raise ValueError("Invalid number of kept samples: {}".format(num_kept_samples))                  
        
        state = self.state[self._params[0]]

        # Define deterministic objective function closure
        seed = time.time()
        def obj_closure_deterministic():
            with opt_utils.random_seed_torch(int(seed)):
                return obj_closure()

        # Load optimization settings
        group = self.param_groups[0]
        gamma = group['gamma']
        dtype = group['dtype']
        verbose = group['verbose']  
        
        # Set objective function closure evaluation counter to zero
        state['obj_closure_eval_count'] = 0
        
        # Start EMNA iteration
        state['n_iter'] += 1        
        mu = self._gather_flat_params()
        Sigma = state.get('Sigma')
        
        # Draw new samples from current sampling distribution
        d = self._calc_numel()
        X = torch.zeros(num_new_samples, d, device=self._device)
        F = torch.zeros(num_new_samples, device=self._device)
        L = torch.linalg.cholesky(Sigma)
        
        for n in range(num_new_samples):
            v_raw = torch.randn(d, device=self._device)
            X[n,:] = L @ v_raw + mu
            F[n] = self._obj_point_evaluate(obj_closure_deterministic, X[n,:])
            
        # Sort samples by objective function closure values
        F_sorted, ind_sorted = torch.sort(F)
        obj_min = F_sorted[0]
        
        # Update sampling distribution mean
        mu = torch.mean(X[ind_sorted[0:num_kept_samples],:], axis=0)
        
        # Update sampling distribution covariance
        S = torch.zeros(d,d, device=self._device)
        x_i = torch.zeros(d, device=self._device)
        for n_k in range(num_kept_samples):
            x_i = X[ind_sorted[n_k],:]
            dis = torch.zeros(d,1, device=self._device)
            dis[:,0] = x_i - mu
            S += dis @ dis.T
            
        Sigma = (1/(num_kept_samples - 1))*S + gamma*torch.eye(d, device=self._device)

        # Set parameters to mean of sampling distribution
        self._update_params_to_point(mu)

        # Update state        
        state['Sigma'] = Sigma
        state['prev_obj_min'] = obj_min
        
        return obj_min
        

