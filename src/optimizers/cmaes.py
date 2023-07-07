# Standard library imports
import time
from functools import reduce

# Third party imports
import cma
import torch
import torch.optim as optim

# Local imports
from . import opt_utils


class CMAES(optim.Optimizer):
    """
    Implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm as a PyTorch 
    optimizer. Based on the CMA-ES implementation available at https://github.com/CMA-ES/pycma.
    
    .. warning::
        This optimizer doesn't support per-parameter options and parameter groups (there can only be one group).
        
    .. warning::
        Right now all parameters have to be on a single device.
    
    Args:
        params (list): list of torch.Tensor objects containing parameters to optimize. 
        population_size (int): CMA-ES population size (default: 5).
        sigma_0 (float): initial CMA-ES step size (default: 1.0).
        active (bool): use the active version of CMA-ES (default: True).
        dtype (dtype or string): the desired data type (default: torch.float).
        verbose (bool): print additional information, useful for debugging (default: False). 
    """
    def __init__(self,
                 params,
                 population_size=5,
                 sigma_0=1.0,
                 active=True,
                 dtype=torch.float,
                 verbose=False):

        defaults = dict(population_size=population_size,
                        sigma_0=sigma_0,
                        active=active,
                        dtype=dtype,
                        verbose=verbose)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("CMA-ES doesn't support per-parameter options (parameter groups)")
            
        self._params = self.param_groups[0]['params']
        self._device = self._params[0].device
        self._numel_cache = None
        self._cma_es = cma.CMAEvolutionStrategy(self._gather_flat_params().cpu().detach(),
                                                sigma_0, 
                                                {'popsize':population_size, 'CMA_active':active})
        
        # NOTE: CMA-ES has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict   
        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0) # iteration count     
        state.setdefault('prev_obj', None) # previous objective function value
        state.setdefault('obj_closure_eval_count', 0) # count of objective function closure evaluations at current iteration
        
        
    def _calc_numel(self):
        """
        Calculate the number of elements to be optimized and cache result. 
        
        Returns:
            _numel_cache (int): total number of elements to be optimized.
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
             obj_closure):
        """
        Performs a single Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization step.
        
        Args:
            obj_closure (callable): A closure that evaluates the objective function.
            
        Returns:     
            obj (float): objective function value of best solution. 
        """
        
        state = self.state[self._params[0]]

        # Define deterministic objective function closure
        seed = time.time()
        def obj_closure_deterministic():
            with opt_utils.random_seed_torch(int(seed)):
                return obj_closure()

        # Load optimization settings
        group = self.param_groups[0]
        population_size = group['population_size']
        sigma_0 = group['sigma_0']
        active = group['active']
        dtype = group['dtype']
        verbose = group['verbose']             
        
        # Set objective function closure evaluation counter to zero
        state['obj_closure_eval_count'] = 0
        
        # Start CMA-ES iteration
        state['n_iter'] += 1

        # Draw new samples from current sampling distribution
        next_points = self._cma_es.ask()
        N = len(next_points)
        F = torch.zeros(N, device=self._device)

        for n in range(N):
            F[n] = self._obj_point_evaluate(obj_closure_deterministic, 
                                            torch.from_numpy(next_points[n]))
        
        self._cma_es.tell(next_points, 
                          F.tolist())

        # Record lowest objective function value obtained so far
        obj = self._cma_es.result[1]
        
        # Set parameters to mean of sampling distribution
        self._update_params_to_point(torch.from_numpy(self._cma_es.result[5]))

        # Update state        
        state['prev_obj'] = obj
        
        return obj
                          

