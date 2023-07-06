# Third party imports
import torch
import torch.nn as nn


class RESNETEULER(nn.Module):
    """
    Type of Residual Neural Network (RESNET) surrogate model with Forward Euler time integration.
    
    Args:
        d_in (int): dimension of input data.
        d_layer (int): dimension of network layers.
        d_out (int): dimension of output.
        num_square_layers (int): number of square network layers.
        dt (float): time step size.
        sigma (float): scaling of inital network weights (default: 1e-3).
        dtype (dtype or string): the desired data type (default: torch.float).
    """
    def __init__(self,
                 d_in,
                 d_layer,
                 d_out,
                 num_square_layers,
                 dt,
                 sigma=1e-3,
                 dtype=torch.float):
                 
        super().__init__()
        
        self.K_open = nn.Parameter(torch.randn(d_layer, d_in, dtype=dtype)*sigma)
        self.b_open = nn.Parameter(torch.ones(d_layer, 1, dtype=dtype)*sigma)
        self.K = nn.Parameter(torch.randn(d_layer, d_layer, num_square_layers, dtype=dtype)*sigma)
        self.b = nn.Parameter(torch.randn(d_layer, 1, num_square_layers, dtype=dtype)*sigma)
        self.K_close = nn.Parameter(torch.randn(d_out, d_layer)*sigma)
        self.b_close = nn.Parameter(torch.ones(d_out, 1, dtype=dtype)*sigma)
        self.dt = dt
        
    
    def forward(self,
                X):
        """
        Forward propagate the input data X through the network.
        
        Args:
            X (torch.Tensor): (d_in) x (# data points) tensor of input data.
            
        Returns:
            y_out (torch.Tensor): (d_out) x (# data points) tensor of output.
        """
        (d_out, d_layer) = self.K_close.shape
        num_square_layers = self.K.shape[2]
        y = torch.relu(self.K_open @ X + self.b_open)
        
        for j in range(num_square_layers):
            K_j = self.K[:,:,j]
            b_j = self.b[:,:,j]
            y = y + self.dt*torch.relu(K_j @ y + b_j)
        
        y_out = self.K_close @ y + self.b_close
        
        return y_out
        
    
    def regularization(self,
                       regularization_type):
        """
        Calculate a regularization term based on the network parameters. 
        
        Args:
            regularization_type (string): "L2-All", "L2-Square" or "L2-Close" to apply L2 (i.e. weight decay) 
                                           regularization to all network layers, only the square layers, or only the
                                           closing layer respectively.
        """
        if regularization_type in ["L2-All", "L2-all"]:
            out_K_open = torch.sum(torch.pow(self.K_open, 2))
            out_b_open = torch.sum(torch.pow(self.b_open, 2))
            out_K = torch.sum(torch.pow(self.K, 2))
            out_b = torch.sum(torch.pow(self.b, 2))
            out_K_close = torch.sum(torch.pow(self.K_close, 2))
            out_b_close = torch.sum(torch.pow(self.b_close, 2))
            out = out_K_open + out_b_open + out_K + out_b + out_K_close + out_b_close
        
        
        elif regularization_type in ["L2-Square", "L2-square"]:
            out_K = torch.sum(torch.pow(self.K, 2))
            out_b = torch.sum(torch.pow(self.b, 2))
            out = out_K + out_b
        
        
        elif regularization_type in ["L2-Close", "L2-close"]:
            out_K_close = torch.sum(torch.pow(self.K_close, 2))
            out_b_close = torch.sum(torch.pow(self.b_close, 2))
            out = out_K_close + out_b_close
        
        
        else:
            raise ValueError("The regularization type %s is not implemented..." % regularization_type)
        
        
        return out            
        
    
    def set_params(self,
                   K_open,
                   b_open,
                   K,
                   b,
                   K_close,
                   b_close,
                   dt):
        """
        Set the parameters of this RESNETEULER to the provided values. 
        
        Args:
            K_open (torch.Tensor): (d_layer) x (d_in) tensor for changing dimensions of feature space.
            b_open (torch.tensor): (d_layer) x 1 bias term for K_open.
            K (torch.tensor): (d_layer) x (d_layer) x (num_square_layers) tensor of kernels to propagate.
            b (torch.tensor): (d_layer) x 1 x (num_square_layers) tensor of biases.
            K_close (torch.tensor): (d_out) x (d_layer) tensor for classification.
            b_close (torch.tensor): (d_out) x 1 bias term for K_close.
            dt (float): time step size.
        """
        self.K_open = K_open
        self.b_open = b_open
        self.K = K
        self.b = b
        self.K_close = K_close
        self.b_close = b_close
        self.dt = dt
        
        
