# Third party imports
import torch
import torch.autograd as autograd
import torch.nn as nn


class CUTEstFunction(autograd.Function):
    """
    Implements PyTorch autograd functionality for CUTEst problems. Uses the PyCUTEst python interface to the CUTEst 
    optimization test environment (see https://jfowkes.github.io/pycutest).
    """
    @staticmethod
    def forward(ctx, 
                input, 
                problem, 
                dtype=torch.float):
        """
        Forward pass. 
        
        Args:
            input (torch.Tensor): point to evaluate the CUTEst objective function at. 
            problem (pycutest.problem_class.CUTEstProblem): PyCUTEst problem.
            dtype (dtype or string): the desired data type (default: torch.float).
        
        Returns:
            torch_obj (torch.Tensor): CUTEst problem objective function value.
        """
        x = input.cpu().clone().detach().numpy()
        obj, grad = problem.obj(x, 
                                gradient=True)        
        ctx.save_for_backward(torch.tensor(grad, 
                                           dtype=dtype))
                                           
        torch_obj = torch.tensor(obj, 
                                 dtype=dtype)
        return torch_obj
        
    
    @staticmethod
    def backward(ctx, 
                 grad_ouput):
        """
        Backward pass. 
        """
        grad, = ctx.saved_tensors
        return grad, None


class CUTEstFunctionAdditiveNoise(autograd.Function):
    """
    Implements PyTorch autograd functionality for CUTEst problems with additive noise. Uses the PyCUTEst python 
    interface to the CUTEst optimization test environment (see https://jfowkes.github.io/pycutest).
    """
    @staticmethod
    def forward(ctx, 
                input, 
                problem, 
                num_samples=1,
                obj_noise_func=None,
                grad_noise_func=None,
                dtype=torch.float):
        """
        Forward pass. 
        
        Args:
            input (torch.Tensor): point to evaluate the CUTEst objective function at. 
            problem (pycutest.problem_class.CUTEstProblem): PyCUTEst problem.
            num_samples (int): number of samples over which to average function and gradient noise. 
            obj_noise_func (callable): Argumentless function returning a single sample of function noise.
            grad_noise_func (callable): Argumentless function returning a single sample of gradient noise.
            dtype (dtype or string): the desired data type (default: torch.float).
        
        Returns:
            torch_obj (torch.Tensor): CUTEst problem objective function value with noise added.
        """
        x = input.cpu().clone().detach().numpy()
        obj_true, grad_true = problem.obj(x, 
                                          gradient=True)
        
        # Add noise to objective function 
        if obj_noise_func is None:
            obj = obj_true
        else:
            obj_sum = 0
            for i in range(num_samples):
                obj_sum += obj_true + obj_noise_func()
                
            obj = obj_sum/num_samples
            
        # Add noise to gradient
        true_grad_tensor = torch.tensor(grad_true, 
                                        dtype=dtype)
        if grad_noise_func is None:
            grad = true_grad_tensor
        else:
            grad_sum = torch.zeros(true_grad_tensor.shape, 
                                   dtype=dtype)
            for i in range(num_samples):
                grad_sum += true_grad_tensor + torch.tensor(grad_noise_func(), 
                                                            dtype=dtype)
                
            grad = grad_sum/num_samples
        
        ctx.save_for_backward(grad)
        
        torch_obj = torch.tensor(obj, 
                                 dtype=dtype)
        return torch_obj
        
        
    @staticmethod
    def backward(ctx, 
                 grad_ouput):
        """
        Backward pass. 
        """
        grad, = ctx.saved_tensors
        return grad, None, None, None, None
        

class CUTEstProblem(nn.Module):
    """
    Implements CUTEst problem as a PyTorch module. Uses the PyCUTEst python interface to the CUTEst optimization test 
    environment (see https://jfowkes.github.io/pycutest).
    
    Args:
        problem (pycutest.problem_class.CUTEstProblem): PyCUTEst problem. 
        dtype (dtype or string): the desired data type (default: torch.float).
    """
    def __init__(self, 
                 problem,
                 dtype=torch.float):
        super().__init__()     
                    
        x = torch.tensor(problem.x0, 
                         dtype=dtype)
        x.requires_grad_()
        
        self.dtype = dtype
        self.variables = torch.nn.Parameter(x)
        self.problem = problem
        
        
    def forward(self):
        """
        Forward pass. 
        
        Returns:
            (torch.Tensor): CUTEst problem objective function value evaluated at current parameters. 
        """
        return CUTEstFunction.apply(self.variables, 
                                    self.problem)
        
        
    def get_obj(self, 
                x):
        """
        Evaluate CUTEst objective function at given point x.
        
        Args:
            x (torch.Tensor): point to evaluate objective function at.
            
        Returns:
            obj (float): value of objective function at x.
        """
        x = x.cpu().clone().detach().numpy()
        obj = self.problem.obj(x, 
                               gradient=False)
        return obj        


    def get_grad(self, 
                 x):
        """
        Evaluate gradient of CUTEst objective function at given point x.
        
        Args:
            x (torch.Tensor): point to evaluate gradient of objective function at.
            
        Returns:    
            torch_grad (torch.Tensor): gradient of objective function at x.
        """
        x = x.cpu().clone().detach().numpy()
        obj, numpy_grad = self.problem.obj(x, 
                                           gradient=True)
        torch_grad = torch.tensor(numpy_grad, 
                                  dtype=self.dtype)
        return torch_grad 
        
        
    def get_hessian(self, 
                    x):
        """
        Evaluate Hessian of CUTEst objective function at given point x.
        
        Args:
            x (torch.Tensor): point to evaluate Hessian of objective function at.
        
        Returns
            torch_H (torch.Tensor): Hessian of objective function at x.
        """
        x = x.cpu().clone().detach().numpy()
        numpy_H = self.problem.hess(x)
        torch_H = torch.tensor(numpy_H, 
                               dtype=self.dtype)
        return torch_H         
        
        
    def get_tensor_x(self):
        """
        Get current x parameters of this CUTEstProblem.
        
        Returns:
            torch_x (torch.Tensor): current x parameters of this CUTEstProblem.
        """
        torch_x = self.variables
        return torch_x
        
        
    def get_tensor_grad(self):
        """
        Get current parameter gradients of this CUTEstProblem.
        
        Returns:
            torch_grad (torch.Tensor): current parameter gradients of this CUTEstProblem.
        """
        torch_grad = self.variables.grad
        return torch_grad        
        
                  
class CUTEstProblemAdditiveNoise(nn.Module):
    """
    Implements CUTEst problem with additive noise as a PyTorch module. Uses the PyCUTEst python interface to the CUTEst 
    optimization test environment (see https://jfowkes.github.io/pycutest).
    
    Args:
        problem (pycutest.problem_class.CUTEstProblem): PyCUTEst problem. 
        obj_noise_func (callable): Argumentless function returning a single sample of function noise.
        grad_noise_func (callable): Argumentless function returning a single sample of gradient noise.
        dtype (dtype or string): the desired data type (default: torch.float).
    """
    def __init__(self, 
                 problem,
                 obj_noise_func,
                 grad_noise_func,
                 dtype=torch.float):
        super().__init__()     
                    
        x = torch.tensor(problem.x0, 
                         dtype=dtype)
        x.requires_grad_()
        
        self.dtype = dtype
        self.variables = torch.nn.Parameter(x)
        self.problem = problem
        self.obj_noise_func = obj_noise_func
        self.grad_noise_func = grad_noise_func
        
        
    def forward(self, 
                num_samples):
        """
        Forward pass. 
        
        Args:
            num_samples (int): number of samples over which to average function and gradient noise. 
            
        Returns:
            (torch.Tensor): CUTEst problem objective function value evaluated at current parameters, with noise added. 
        """
        return CUTEstFunctionAdditiveNoise.apply(self.variables, 
                                                 self.problem,
                                                 num_samples,
                                                 self.obj_noise_func,
                                                 self.grad_noise_func)
        

    def get_true_obj(self, 
                     x):
        """
        Evaluate true (i.e. non-noisy) CUTEst objective function at given point x.
        
        Args:
            x (torch.Tensor): point to evaluate objective function at.
            
        Returns:
            obj (float): value of objective function at x without noise.
        """
        x = x.cpu().clone().detach().numpy()
        obj = self.problem.obj(x, 
                               gradient=False)
        return obj
        
        
    def get_true_grad(self, 
                      x):
        """
        Evaluate true (i.e. non-noisy) gradient of CUTEst objective function at given point x.
        
        Args:
            x (torch.Tensor): point to evaluate gradient of objective function at.
            
        Returns:    
            torch_grad (torch.Tensor): gradient of objective function at x without noise.
        """
        x = x.cpu().clone().detach().numpy()
        obj, numpy_grad = self.problem.obj(x, 
                                           gradient=True)
        torch_grad = torch.tensor(numpy_grad, 
                                  dtype=self.dtype)
        return torch_grad        


    def get_true_hessian(self, 
                         x):
        """
        Evaluate true (i.e. non-noisy) Hessian of CUTEst objective function at given point x.
        
        Args:
            x (torch.Tensor): point to evaluate Hessian of objective function at.
        
        Returns
            torch_H (torch.Tensor): Hessian of objective function at x without noise.
        """
        x = x.cpu().clone().detach().numpy()
        numpy_H = self.problem.hess(x)
        torch_H = torch.tensor(numpy_H, 
                               dtype=self.dtype)
        return torch_H  
        
        
    def get_x(self):
        """
        Get current x parameters of this CUTEstProblemAdditiveNoise.
        
        Returns:
            x (torch.Tensor): current x parameters of this CUTEstProblemAdditiveNoise.
        """
        x = self.variables
        return x
        
        
    def get_grad(self):
        """
        Get current (noisy) parameter gradients of this CUTEstProblemAdditiveNoise.
        
        Returns:
            grad (torch.Tensor): current (noisy) parameter gradients of this CUTEstProblemAdditiveNoise.
        """
        grad = self.variables.grad
        return grad
        

