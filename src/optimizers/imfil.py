# Standard library imports
import time
from functools import reduce

# Third party imports
import torch
import torch.optim as optim

# Local imports
from . import opt_utils


class IMFIL(optim.Optimizer):
    """
    Implements the Implicit Filtering (IF) optimization algorithm as a PyTorch optimizer. For a comprehensive reference 
    regarding implicit filtering, see the book "Implicit Filtering" by C. T. Kelley at the link
    http://dx.doi.org/10.1137/1.9781611971903.
    
    .. warning::
        This optimizer doesn't support per-parameter options and parameter groups (there can only be one group).
        
    .. warning::
        Right now all parameters have to be on a single device.

    Args:
        params (list): list of torch.Tensor objects containing parameters to optimize. 
        h_0 (float): initial stencil size (default: 1.0). 
        hessian_approx (string): Hessian approximation - either "I" for identity matrix or "BFGS" (default: "I").
        H_0 (torch.Tensor, optional): initial inverse Hessian approximation.
        dtype (dtype or string): the desired data type (default: torch.float).
        verbose (bool): print additional information, useful for debugging (default: False).            
    """
    def __init__(self, 
                 params,
                 h_0=1.0,
                 hessian_approx="I",
                 H_0=None,
                 dtype=torch.float,
                 verbose=False):

        # Check that initial stencil size is valid 
        if not h_0 > 0.0:
            raise ValueError("Invalid initial stencil size (must be positive): {}".format(h_0))
            
        # Check that Hessian approximation mode is valid
        if hessian_approx not in ["I", "BFGS"]:
            raise ValueError("Invalid Hessian approximation mode: {}".format(hessian_approx))
            
        defaults = dict(h_0=h_0,
                        hessian_approx=hessian_approx,
                        dtype=dtype,
                        verbose=verbose)
        super().__init__(params, defaults)
        
        if len(self.param_groups) != 1:
            raise ValueError("Implicit filtering doesn't support per-parameter options (parameter groups)")       
            
        self._params = self.param_groups[0]['params']
        self._device = self._params[0].device
        self._numel_cache = None
        
        # NOTE: IF has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict   
        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0) # iteration count
        state.setdefault('prev_flat_stencil_grad', None) # previous flat stencil gradient
        state.setdefault('prev_obj', None) # previous best objective function value
        state.setdefault('prev_V', None) # previous directions 
        state.setdefault('prev_F', None) # previous objective function closure evaluations
        state.setdefault('obj_closure_eval_count', 0) # count of objective function closure evaluations at current iteration
        state.setdefault('line_search_obj_closure_eval_count', 0) # current iteration count of objective function evaluations for line search
        state.setdefault('h', h_0) # stencil size
        state.setdefault('stencil_failure', False) # flag for stencil failure
        state.setdefault('H', H_0.to(device=self._device)) # Hessian approximation 
        
    
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
        
    
    def _update_params(self, 
                       step_size, 
                       flat_update_direction):
        """
        Update parameters being optimized by taking a step in given update direction.
        
        Args:
            step_size (float): how far to move along update direction.
            flat_update_direction (torch.Tensor): one dimensional tensor containing update direction.
        """
        offset = 0
        for p in self._params:
            numel = p.numel()
            # use view to avoid deprecated pointwise semantics
            p.add_(flat_update_direction[offset:offset + numel].view_as(p), alpha=step_size)
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
                        
                        
    def _obj_directional_evaluate(self,
                                  obj_closure,
                                  step_size,
                                  flat_direction):
        """
        Evaluate the objective function along given direction.
        
        Args:
            obj_closure (callable): A closure that evaluates the objective function.
            step_size (float): how far to evaluate along given direction.
            flat_direction (torch.Tensor): one dimensional tensor containing direction to evaluate along. 
            
        Returns:
            obj (float): Value of objective function at point along given direction.
        """
        state = self.state[self._params[0]]        
        
        # store initial point in parameter space
        x = self._clone_param() 
        
        # take a step in parameter space
        self._update_params(step_size, 
                            flat_direction)
        
        with torch.no_grad():
            # evaluate objective function only
            obj = obj_closure()
            state['obj_closure_eval_count'] += 1
            
            # restore parameters to initial values
            self._set_param(x) 
                
            obj = float(obj)
                
            return obj
            
    
    def _get_stencil_directions(self,
                                stencil_type):
        """
        Get stencil directions.
        
        Args:
            stencil_type (string): stencil type - either "CD" for central finite difference stencil (size 2d), "a-PBS" 
                                   for an asymmetric positive basis stencil (size d+1), "FD" for forward finite 
                                   difference stencil (size d), or "None". 
        
        Returns:
            V_stencil (torch.Tensor): matrix of stencil directions.
        """
        d = self._calc_numel()
        
        # Get stencil directions
        if stencil_type in ["CD"]:
            V_pos = torch.eye(d, device=self._device)
            V_neg = -V_pos
            V_stencil = torch.cat((V_pos, V_neg), dim=1)
            
        elif stencil_type in ["a-PBS"]:
            V_pos = torch.eye(d, device=self._device)
            V_stencil = torch.cat((V_pos, (-1/(d**(1/2)))*torch.ones(d, device=self._device).unsqueeze(1)), dim=1)
            
        elif stencil_type in ["FD"]:
            V_pos = torch.eye(d, device=self._device)
            V_stencil = V_pos
            
        elif stencil_type in ["None"]:
            V_stencil = torch.tensor([], device=self._device)
            
        else:
            raise ValueError("The stencil type %s is not implemented ..." % stencil_type)
            
        return V_stencil
    
    
    def _get_custom_sampler_directions(self,
                                       custom_sampler,
                                       num_dirs_custom):
        """
        Get custom sampler directions.
        
        Args:
            custom_sampler (callable): argumentless function that generates d-dimensional vectors.
            num_dirs_custom (int): number of directions to generate using custom_sampler.
            
        Returns:
            V_sampled (torch.Tensor): matrix of custom sampler directions.
        """
        d = self._calc_numel()
        
        # Get custom sampler directions
        if num_dirs_custom > 0 and custom_sampler is not None:
            
            V_sampled = torch.zeros(d, num_dirs_custom, device=self._device)
            
            # Draw new directions from custom sampler
            for n in range(num_dirs_custom):
                V_sampled[:,n] = custom_sampler()
                
        else:
            V_sampled = torch.tensor([], device=self._device)
            
        return V_sampled
    
    
    def _get_stencil_gradient(self,
                              V,
                              h,
                              F_diffs,
                              beta_grad):
        """
        Compute stencil gradient.
        
        Args:
            V (torch.Tensor): stencil directions.
            h (float): stencil size.
            F_diffs (torch.Tensor): differences of objective function closure with respect to center point. 
            beta_grad (float): regularization parameter to use when solving for stencil gradient.
            
        Returns:
            flat_stencil_grad (torch.Tensor): stencil gradient stored as one dimensional tensor.
        """
        d = self._calc_numel()
        num_dirs_total = V.shape[1]
        
        if d <= num_dirs_total:
            
            # Use closed form solution for overdetermined linear system with L2 regularization
            flat_stencil_grad = (1/h)*torch.linalg.solve((V@(V.T) + beta_grad*torch.eye(V.shape[0], device=self._device)), V@F_diffs)
            
        else:
            
            # Use closed form solution for underdetermined linear system with L2 regularization 
            flat_stencil_grad = (1/h)*V@torch.linalg.solve(((V.T)@V + beta_grad*torch.eye(V.shape[1], device=self._device)), F_diffs)
            
        return flat_stencil_grad
        
        
    def _calc_BFGS_inverse_Hessian_update(self,
                                          y,
                                          s,
                                          H):
        """
        Calculates the BFGS inverse Hessian update formula if the BFGS curvature condition is satisfied.
        
        Args:
            y (torch.Tensor): difference of stencil gradients.
            s (torch.Tensor): difference of iterates.
            H (torch.Tensor): current inverse Hessian approximation.
        
        Returns:
            H_new (torch.Tensor): BFGS updated inverse Hessian approximation.
        """
        d = self._calc_numel()
        
        y_dot_s = y.dot(s)
        
        # Check BFGS curvature condition
        if y_dot_s > 0:
            
            I = torch.eye(d, device=self._device)
            rho = 1/(y_dot_s)
            s_outer_product_y_T = s.unsqueeze(1)@(y.unsqueeze(1).T)
            s_outer_product_s_T = s.unsqueeze(1)@(s.unsqueeze(1).T)
            H_new = (I - rho*s_outer_product_y_T)@H@(I - rho*s_outer_product_y_T.T) + rho*s_outer_product_s_T
            
        else:
            H_new = H
            
        return H_new
    
    
    def _check_line_search_decrease_condition(self,
                                              step_size,
                                              obj,
                                              obj_try, 
                                              tau):
        """
        Check line search decrease condition, backtracking if the condition fails.
        
        Args:
            step_size (float): current step size.
            obj (float): objective at original/starting point.
            obj_try (float): objective at new point to try (i.e. proposed point).
            tau (float): backtracking factor 0 < tau < 1.
            
        Returns:
            found (bool): true if decrease condition is satisfied, false otherwise.
            step_size (float): step size after applying backtracking (backtracking only occurs if found is false).
        """
        found = False
        
        break_condition = obj_try - obj
        
        if (break_condition < 0):
            found = True
        
        else:
            # Decrease step size via backtracking
            step_size = step_size*tau
        
        return found, step_size 
    
    
    @torch.no_grad()
    def step(self,
             obj_closure,
             stencil_type,
             tau_tr=0.5,
             tau_grad=1e-2,
             step_size_ls=1e0,
             max_line_search_obj_closure_evals=3,
             tau_ls=0.1,
             stencil_wins=False,
             custom_sampler=None,
             num_dirs_custom=0,
             beta_grad=1e-5):    
        """
        Performs a single IF optimization step.
        
        Args:
            obj_closure (callable): A closure that evaluates the objective function.
            stencil_type (string): Stencil type - either "CD" for central finite difference stencil (size 2d), "a-PBS" 
                                   for an asymmetric positive basis stencil (size d+1), "FD" for forward finite 
                                   difference stencil (size d), or "None" (use "None" to rely on only custom_sampler). 
            tau_tr (float): factor in (0,1) by which to shrink h upon failure (default: 0.5).
            tau_grad (float): factor by which to compare stencil gradient norm relative to h (default: 1e-2).
            step_size_ls (float): initial line search step size (default: 1e0).
            max_line_search_obj_closure_evals (int): maximum number of objective function closure evaluations for line search (default: 3).
            tau_ls (float): backtracking factor in (0,1) (default: 0.1).
            stencil_wins (bool): take stencil point if it is better than the line search point (default: false).
            custom_sampler (callable, optional): argumentless function that generates d-dimensional vectors.
            num_dirs_custom (int): number of directions to generate using custom_sampler (default: 0).
            beta_grad (float): regularization parameter to use when solving for stencil gradient (default: 1e-5).
            
        Returns:     
            obj (float): objective function value of best solution. 
        """
        # Check that stencil type is valid
        if stencil_type not in ["CD", "a-PBS", "FD", "None"]:
            raise ValueError("Invalid stencil type: {}".format(stencil_type))
        
        # Check that tau_tr is valid
        if tau_tr <= 0.0 or tau_tr >= 1.0:
            raise ValueError("Invalid tau_tr: {}".format(tau_tr))
        
        # Check that number of custom_sampler directions is valid
        if not num_dirs_custom >= 0:
            raise ValueError("Invalid number of custom sampler directions: {}".format(num_dirs_custom))
        
        # Check that initial line search step size is valid
        if not step_size_ls > 0.0:
            raise ValueError("Invalid initial line search step size: {}".format(step_size_ls)) 
        
        state = self.state[self._params[0]]
        state['step_size'] = step_size_ls
        
        # Define deterministic objective function closure
        seed = time.time()
        def obj_closure_deterministic():
            with opt_utils.random_seed_torch(int(seed)):
                return obj_closure()
        
        # Load optimization settings
        group = self.param_groups[0]
        h_0 = group['h_0']
        hessian_approx = group['hessian_approx']
        dtype = group['dtype']        
        verbose = group['verbose']
        
        # Set objective function closure evaluation counters to zero
        state['obj_closure_eval_count'] = 0
        state['line_search_obj_closure_eval_count'] = 0
        
        # Start IF iteration 
        state['n_iter'] += 1 
        h = state.get('h')
        
        # Store current point
        prev_flat_params = self._gather_flat_params()
        
        # Evaluate f(x) and stencil gradient if first iteration
        if state['n_iter'] == 1:
            with torch.no_grad():
                obj = float(obj_closure_deterministic())
                state['obj_closure_eval_count'] += 1
            
            # Store initial objective function closure value
            state['prev_obj'] = obj
            
            # Prepare to get directions
            V = torch.tensor([], device=self._device)
            
            # Get stencil directions
            V_stencil = self._get_stencil_directions(stencil_type=stencil_type)
            
            # Get custom sampler directions
            V_sampled = self._get_custom_sampler_directions(custom_sampler=custom_sampler,
                                                            num_dirs_custom=num_dirs_custom)
            
            # Store stencil and custom sampler directions
            V = torch.cat((V, V_stencil, V_sampled), dim=1)
            
            # Prepare to evaluate objective function closure at all points 
            F = torch.zeros(V.shape[1], device=self._device)
            
            # Evaluate objective function closure at all points 
            for n in range(V.shape[1]):
                F[n] = self._obj_directional_evaluate(obj_closure_deterministic,
                                                      h,
                                                      V[:,n])
            
            # Store directions and corresponding objective function closure evaluations
            state['prev_V'] = V
            state['prev_F'] = F
            
            # Compute initial stencil gradient 
            F_diffs = F - obj
            flat_stencil_grad = self._get_stencil_gradient(V=V,
                                                           h=h,
                                                           F_diffs=F_diffs,
                                                           beta_grad=beta_grad)
            
            # Store initial stencil gradient
            state['prev_flat_stencil_grad'] = flat_stencil_grad
            
        
        prev_best_obj = state.get('prev_obj')
        V = state.get('prev_V')
        F = state.get('prev_F')
        prev_flat_stencil_grad = state.get('prev_flat_stencil_grad')
        
        # Find minimum of all objective function closure values
        f_best, ind_best = torch.min(F, 
                                     dim=0)  
        
        # Check for stencil failure
        if f_best >= prev_best_obj:
            state['stencil_failure'] = True 
        else:
            state['stencil_failure'] = False
            f_try_stencil = f_best
            ind_try_stencil = ind_best 
            
        
        if state['stencil_failure'] is True or torch.linalg.norm(prev_flat_stencil_grad) <= tau_grad*h:
            # Shrink trust region size
            h = tau_tr*h
        
            if verbose:
                print("Reducing h to: {}".format(h))
                
        else:
            if hessian_approx in ["I"]:
                flat_update_direction = - prev_flat_stencil_grad
                
            elif hessian_approx in ["BFGS"]:
                H = state.get('H')
                flat_update_direction = - H@prev_flat_stencil_grad
            
            # Line search
            step_size = state.get('step_size')
            f_try_ls = self._obj_directional_evaluate(obj_closure_deterministic,
                                                      step_size,
                                                      flat_update_direction)
            state['line_search_obj_closure_eval_count'] += 1
            
            # Check if new point decreases objective
            found, step_size = self._check_line_search_decrease_condition(step_size=step_size,
                                                                          obj=prev_best_obj,
                                                                          obj_try=f_try_ls, 
                                                                          tau=tau_ls)
            
            while found is False or not opt_utils.is_legal(torch.tensor(f_try_ls, dtype=dtype)):
                if verbose:
                    print("Obj. closure evaluation count: {}".format(state['obj_closure_eval_count']))
                    print("Step size: {}".format(step_size))
                    print("New f: {}".format(f_try_ls))
                    print("Old f: {}".format(prev_best_obj))
                    print("Update direction norm: {}".format(torch.linalg.norm(flat_update_direction)))
                    
                if state['line_search_obj_closure_eval_count'] < max_line_search_obj_closure_evals:
                    # Try a new point
                    f_try_ls = self._obj_directional_evaluate(obj_closure_deterministic,
                                                              step_size,
                                                              flat_update_direction)
                    state['line_search_obj_closure_eval_count'] += 1
                    found, step_size = self._check_line_search_decrease_condition(step_size=step_size,
                                                                                  obj=prev_best_obj,
                                                                                  obj_try=f_try_ls, 
                                                                                  tau=tau_ls)
                                                                                  
                else:
                    step_size = 0
                    break
                    
            
            # Select new point 
            if found is False:
                # Take the stencil point if line search fails 
                prev_best_obj = f_try_stencil
                self._update_params(h,
                                    V[:,ind_try_stencil])
                
            else:
            
                if stencil_wins is True and f_try_stencil < f_try_ls:
                    # Take the stencil point
                    prev_best_obj = f_try_stencil
                    self._update_params(h,
                                        V[:,ind_try_stencil])
                
                else:
                    # Take the line search point
                    prev_best_obj = f_try_ls
                    self._update_params(step_size,
                                        flat_update_direction)
                    
                    
        # Prepare to get new directions
        V = torch.tensor([], device=self._device)
        
        # Get stencil directions
        V_stencil = self._get_stencil_directions(stencil_type=stencil_type)
        
        # Get custom sampler directions
        V_sampled = self._get_custom_sampler_directions(custom_sampler=custom_sampler,
                                                        num_dirs_custom=num_dirs_custom)
        
        # Store stencil and custom sampler directions
        V = torch.cat((V, V_stencil, V_sampled), dim=1)
        
        # Prepare to evaluate objective function closure at all points 
        F = torch.zeros(V.shape[1], device=self._device)
        
        # Evaluate objective function closure at all points 
        for n in range(V.shape[1]):
            F[n] = self._obj_directional_evaluate(obj_closure_deterministic,
                                                  h,
                                                  V[:,n])
            
        # Store directions and corresponding objective function closure evaluations
        state['prev_V'] = V
        state['prev_F'] = F
        
        # Compute new stencil gradient 
        F_diffs = F - prev_best_obj
        flat_stencil_grad = self._get_stencil_gradient(V=V,
                                                       h=h,
                                                       F_diffs=F_diffs,
                                                       beta_grad=beta_grad)
                                                       
        # Update inverse Hessian approximation if appropriate 
        if not (state['stencil_failure'] is True or torch.linalg.norm(prev_flat_stencil_grad) <= tau_grad*h):    
            if hessian_approx in ["BFGS"]:
                H = state.get('H')
                s = self._gather_flat_params() - prev_flat_params
                y = flat_stencil_grad - prev_flat_stencil_grad
                H_new = self._calc_BFGS_inverse_Hessian_update(y=y,
                                                               s=s,
                                                               H=H)
                state['H'] = H_new
        
        
        # Update state 
        state['h'] = h
        state['prev_flat_stencil_grad'] = flat_stencil_grad
        state['prev_obj'] = prev_best_obj
        obj = prev_best_obj
        
        return obj
        

