# Third party imports
from haven import haven_utils as hu


def get_high_dimensional_CUTEst_benchmark_dict(benchmark_name,
                                               opt_dict_list,
                                               num_samples=1,
                                               max_func_evals=2000,
                                               runs=25):
    """
    Get high dimensional (200 < d <= 3000) CUTEst benchmark tests. 
    
    Args:
        benchmark_name (string): name of benchmark. 
        opt_dict_list (list): list of optimizer dictionaries. 
        num_samples (int): number of samples to average objective function and gradient over (default: 1).
        max_func_evals (int): maximum number of function evaluations (default: 2000).
        runs (int): number of independent optimization runs (default: 25).
        
    Returns:
        benchmark_dict (dict): dictionary storing benchmark information.
    """
    
    # High dimensional (200 < d <= 3000) problems
    # ===============
    if benchmark_name in ["CUTEst No Noise"]:
        cutest_problem_dict_list = [{"CUTEst problem":"ARWHEAD",
                                     "SIF params":{"N":500},
                                     "solution":0.0},
                                    {"CUTEst problem":"BOXPOWER",
                                     "SIF params":{"N":1000},
                                     "solution":0.0},
                                    {"CUTEst problem":"CHAINWOO",
                                     "SIF params":{"NS":499},
                                     "solution":0.0},
                                    {"CUTEst problem":"CYCLOOCFLS",
                                     "SIF params":{"P":1000},
                                     "solution":0.0},
                                    {"CUTEst problem":"EXTROSNB",
                                     "SIF params":{"N":1000},
                                     "solution":0.0},
                                    {"CUTEst problem":"MODBEALE",
                                     "SIF params":{"N/2":1000},
                                     "solution":0.0},
                                    {"CUTEst problem":"SROSENBR",
                                     "SIF params":{"N/2":250},
                                     "solution":0.0}
                                   ]
        noise_type_list = ["additive scaled by starting point"]
        obj_noise_dict_list = [{"name":"symmetric uniform random sampler",
                                "scaling factor":0.0}
                              ]
        grad_noise_dict_list = [{"name":"spherical ball uniform random sampler",
                                 "scaling factor":0.0}
                               ]
        num_samples_list = [num_samples]
        max_func_evals_list = [max_func_evals]
        runs_list = [i for i in range(runs)]
        benchmark_dict = {"CUTEst problem dictionary":cutest_problem_dict_list,
                          "noise type":noise_type_list,
                          "objective noise dictionary":obj_noise_dict_list,
                          "gradient noise dictionary":grad_noise_dict_list,
                          "optimizer dictionary":opt_dict_list,
                          "number of samples":num_samples_list,
                          "maximum number of function evaluations":max_func_evals_list,
                          "runs":runs_list
                         }
    
    
    elif benchmark_name in ["CUTEst Scaled Noise"]:
        cutest_problem_dict_list = [{"CUTEst problem":"ARWHEAD",
                                     "SIF params":{"N":500},
                                     "solution":0.0},
                                    {"CUTEst problem":"BOXPOWER",
                                     "SIF params":{"N":1000},
                                     "solution":0.0},
                                    {"CUTEst problem":"CHAINWOO",
                                     "SIF params":{"NS":499},
                                     "solution":0.0},
                                    {"CUTEst problem":"CYCLOOCFLS",
                                     "SIF params":{"P":1000},
                                     "solution":0.0},
                                    {"CUTEst problem":"EXTROSNB",
                                     "SIF params":{"N":1000},
                                     "solution":0.0},
                                    {"CUTEst problem":"MODBEALE",
                                     "SIF params":{"N/2":1000},
                                     "solution":0.0},
                                    {"CUTEst problem":"SROSENBR",
                                     "SIF params":{"N/2":250},
                                     "solution":0.0}
                                   ]
        noise_type_list = ["additive scaled by starting point"]
        obj_noise_dict_list = [{"name":"symmetric uniform random sampler",
                                "scaling factor":1e-4}
                              ]
        grad_noise_dict_list = [{"name":"spherical ball uniform random sampler",
                                 "scaling factor":0.0}
                               ]
        num_samples_list = [num_samples]
        max_func_evals_list = [max_func_evals]
        runs_list = [i for i in range(runs)]
        benchmark_dict = {"CUTEst problem dictionary":cutest_problem_dict_list,
                          "noise type":noise_type_list,
                          "objective noise dictionary":obj_noise_dict_list,
                          "gradient noise dictionary":grad_noise_dict_list,
                          "optimizer dictionary":opt_dict_list,
                          "number of samples":num_samples_list,
                          "maximum number of function evaluations":max_func_evals_list,
                          "runs":runs_list
                         }
    
    
    else:
        raise ValueError("The high dimensional (200 < d <= 3000) CUTEst benchmark %s is not implemented..." % benchmark_name)
        
        
    return benchmark_dict      


# Optimizer setup
# ===============
CMAES_dict_list = [{"name":"CMA-ES",
                    "alpha":1e0,
                    "population size":30,
                    "sigma_0":1e-1,
                    "active":True,
                    "verbose":False}
                  ]


EMNA_dict_list = [{"name":"EMNA",
                   "alpha":1e0,
                   "sigma_0":1e-1,
                   "gamma":1e-3, 
                   "number of new samples to draw":30,
                   "number of samples to keep":15,
                   "verbose":False}
                 ]


IMFIL_dict_list = [{"name":"IMFIL",
                    "alpha":1e0,
                    "h_0":1e1,
                    "h_min":1e-8,
                    "hessian approximation":"I",
                    "initial inverse hessian":"I",
                    "custom sampler":"spherical surface uniform random sampler",
                    "number of custom sampler directions":30,
                    "stencil type":"None",
                    "tau_tr":0.5,
                    "tau_gr":1e-2,
                    "line search starting step size":1e0,
                    "maximum number of line search objective closure evaluations":3,
                    "tau_ls":0.1,
                    "stencil wins?":True,
                    "stencil gradient beta":1e-5,
                    "verbose":False},
                   {"name":"IMFIL",
                    "alpha":1e0,
                    "h_0":1e1,
                    "h_min":1e-8,
                    "hessian approximation":"BFGS",
                    "initial inverse hessian":"I",
                    "custom sampler":"spherical surface uniform random sampler",
                    "number of custom sampler directions":30,
                    "stencil type":"None",
                    "tau_tr":0.5,
                    "tau_gr":1e-2,
                    "line search starting step size":1e0,
                    "maximum number of line search objective closure evaluations":3,
                    "tau_ls":0.1,
                    "stencil wins?":True,
                    "stencil gradient beta":1e-5,
                    "verbose":False}
                  ]


SGPGD_dict_list = [{"name":"SG-PGD",
                    "alpha":1e0,
                    "h":1e-4,
                    "hessian approximation":"I",
                    "initial inverse hessian":"I",
                    "custom sampler":"spherical surface uniform random sampler",
                    "number of custom sampler directions":30,
                    "stencil type":"None",
                    "line search starting step size":1e0,
                    "maximum number of line search objective closure evaluations":75,
                    "tau_ls":0.1,
                    "stencil gradient beta":1e-5,
                    "Armijo condition c":1e-4,
                    "epsilon_a":0.0,
                    "verbose":False},
                   {"name":"SG-PGD",
                    "alpha":1e0,
                    "h":1e-4,
                    "hessian approximation":"BFGS",
                    "initial inverse hessian":"I",
                    "custom sampler":"spherical surface uniform random sampler",
                    "number of custom sampler directions":30,
                    "stencil type":"None",
                    "line search starting step size":1e0,
                    "maximum number of line search objective closure evaluations":75,
                    "tau_ls":0.1,
                    "stencil gradient beta":1e-5,
                    "Armijo condition c":1e-4,
                    "epsilon_a":0.0,
                    "verbose":False}
                  ]


NNAIF_dict_list = [{"name":"NNAIF",
                    "alpha":1e0,
                    "h_0":1e1,
                    "h_min":1e-8,
                    "hessian approximation":"I",
                    "initial inverse hessian":"I",
                    "custom sampler":"spherical surface uniform random sampler",
                    "number of custom sampler directions":30,
                    "stencil type":"None",
                    "tau_tr":0.5,
                    "tau_gr":1e-2,
                    "h_min^surr":1e-2,
                    "eps_dec^surr":1e-3,
                    "line search starting step size":1e0,
                    "maximum number of line search objective closure evaluations":3,
                    "tau_ls":0.1,
                    "stencil wins?":True,
                    "stencil gradient beta":1e-5,
                    "number of points desired to pass surrogate filter test":10,
                    "maximum number of surrogate filtered sampling iterations":20,
                    "surrogate model dictionary":{"type":"RESNET EULER",
                                                  "d_layer":50,
                                                  "d_out":1,
                                                  "number of square layers":5,
                                                  "dt":1e-3,
                                                  "sigma":1e-3},
                    "output transform dictionary":{"type":"I"},
                    "surrogate fit optimizer dictionary":{"name":"ADAM",
                                                          "alpha":1e0,
                                                          "amsgrad?":True,
                                                          "learning rate":1e-2,
                                                          "betas0":0.9,
                                                          "betas1":0.999,
                                                          "loss function dictionary":{"name":"squared error"},
                                                          "parameter regularization dictionary":{"type":"Model Specific",
                                                                                                 "alpha_x":1e-4,
                                                                                                 "kwargs":{"regularization_type":"L2-All"}},
                                                                                                 "model output regularizer dictionary":{"type":"None"},
                                                          "batch size":100,
                                                          "maximum number of epochs":20,
                                                          "loss tolerance":1e-3},
                                                          "surrogate descent optimizer dictionary":{"name":"ADAM",
                                                                                                    "alpha":1e0,
                                                                                                    "amsgrad?":True,
                                                                                                    "learning rate":1e-1,
                                                                                                    "betas0":0.9,
                                                                                                    "betas1":0.999,
                                                                                                    "maximum number of surrogate descent iterations":20,
                                                                                                    "norm order":None},
                                                                                                    "verbose":False},
                   {"name":"NNAIF",
                    "alpha":1e0,
                    "h_0":1e1,
                    "h_min":1e-8,
                    "hessian approximation":"BFGS",
                    "initial inverse hessian":"I",
                    "custom sampler":"spherical surface uniform random sampler",
                    "number of custom sampler directions":30,
                    "stencil type":"None",
                    "tau_tr":0.5,
                    "tau_gr":1e-2,
                    "h_min^surr":1e-2,
                    "eps_dec^surr":1e-3,
                    "line search starting step size":1e0,
                    "maximum number of line search objective closure evaluations":3,
                    "tau_ls":0.1,
                    "stencil wins?":True,
                    "stencil gradient beta":1e-5,
                    "number of points desired to pass surrogate filter test":10,
                    "maximum number of surrogate filtered sampling iterations":20,
                    "surrogate model dictionary":{"type":"RESNET EULER",
                                                  "d_layer":50,
                                                  "d_out":1,
                                                  "number of square layers":5,
                                                  "dt":1e-3,
                                                  "sigma":1e-3},
                    "output transform dictionary":{"type":"I"},
                    "surrogate fit optimizer dictionary":{"name":"ADAM",
                                                          "alpha":1e0,
                                                          "amsgrad?":True,
                                                          "learning rate":1e-2,
                                                          "betas0":0.9,
                                                          "betas1":0.999,
                                                          "loss function dictionary":{"name":"squared error"},
                                                          "parameter regularization dictionary":{"type":"Model Specific",
                                                                                                 "alpha_x":1e-4,
                                                                                                 "kwargs":{"regularization_type":"L2-All"}},
                                                                                                 "model output regularizer dictionary":{"type":"None"},
                                                          "batch size":100,
                                                          "maximum number of epochs":20,
                                                          "loss tolerance":1e-3},
                                                          "surrogate descent optimizer dictionary":{"name":"ADAM",
                                                                                                    "alpha":1e0,
                                                                                                    "amsgrad?":True,
                                                                                                    "learning rate":1e-1,
                                                                                                    "betas0":0.9,
                                                                                                    "betas1":0.999,
                                                                                                    "maximum number of surrogate descent iterations":20,
                                                                                                    "norm order":None},
                                                                                                    "verbose":False}
                  ]


opt_dict_list = NNAIF_dict_list + IMFIL_dict_list + SGPGD_dict_list + EMNA_dict_list + CMAES_dict_list


# Benchmark setup 
# ===============
HIGH_DIM_CUTEST_EXP_GROUPS = {}
benchmarks_list = ["CUTEst No Noise",
                   "CUTEst Scaled Noise"
                  ]


for benchmark_name in benchmarks_list:
    benchmark_dict = get_high_dimensional_CUTEst_benchmark_dict(benchmark_name,
                                                                opt_dict_list)
    HIGH_DIM_CUTEST_EXP_GROUPS["%s" % benchmark_name] = hu.cartesian_exp_group(benchmark_dict)
    

