# Standard library imports
import argparse
import copy
import math
import os
import pprint
import sys
import time

# Third party imports
import numpy as np
import pandas as pd
import pycutest
import torch
from haven import haven_chk as hc
from haven import haven_utils as hu

# Local imports
import exp_configs
from src import models
from src import optimizers
from src import samplers


def get_surrogate_model_output_transform(output_transform_dict):
    """
    Create callable that transforms surrogate model output to a scalar according to information in output transform
    dictionary (output_transform_dict).
    
    Args:
        output_transform_dict (dict): dictionary storing surrogate model output transform information.
        
    Returns:
        output_transform (callable): transforms surrogate model output to a scalar.
    """
    output_transform_type = output_transform_dict["type"]

    if output_transform_type in ["I", "Identity"]:
        output_transform = lambda x : x
    
    else:
        raise ValueError("The output transform %s is not implemented ..." % output_transform_type)
        
        
    return output_transform


def opt_step(opt_dict, 
             opt, 
             model, 
             num_samples, 
             sampler=None):
    """
    Define appropriate optimization step procedure for various optimizers. 
    
    Args:
        opt_dict (dict): dictionary storing optimizer information.
        opt (torch.optim.Optimizer): PyTorch optimizer.
        model (torch.nn.Module): PyTorch model.
        num_samples (int): number of samples to average CUTEst objective function and gradient over.
        sampler (callable): argumentless function that generates d-dimensional vectors.
        
    Returns:
        obj (float): objective function value of best solution. 
    """
    alpha = opt_dict["alpha"]
    opt_name = opt_dict["name"]
    
    # Zero gradient 
    opt.zero_grad()
    
    def scaled_obj_closure():
        """
        A closure that evaluates the CUTEst problem objective function scaled by 1/alpha.
        """
        obj_fun = model(num_samples)/alpha
        return obj_fun
        
        
    # Custom optimizers
    # ===============
    if opt_name in ["CMA-ES"]:
        obj = opt.step(scaled_obj_closure)
    
    
    elif opt_name in ["EMNA"]:
        num_new_samples = opt_dict["number of new samples to draw"]
        num_kept_samples = opt_dict["number of samples to keep"]
        
        if num_new_samples in ["d"]:
            num_new_samples = sum(p.numel() for p in model.parameters())
            
        elif num_new_samples in ["d+1"]:
            num_new_samples = sum(p.numel() for p in model.parameters()) + 1
            
        if num_kept_samples in ["half"]:
            num_kept_samples = math.ceil(num_new_samples/2)
            
        obj = opt.step(scaled_obj_closure,
                       num_new_samples=num_new_samples,
                       num_kept_samples=num_kept_samples)

    
    elif opt_name in ["IMFIL"]:
        if sampler is None:
            custom_sampler_type = opt_dict["custom sampler"]
            
            if custom_sampler_type not in ["None"]:
                custom_sampler_dict = {"name":custom_sampler_type, 
                                       "dimension":sum(p.numel() for p in model.parameters())}
                custom_sampler = samplers.get_vector_sampler(custom_sampler_dict)
            
            else:
                custom_sampler = None
                
        stencil_type = opt_dict["stencil type"]
        tau_tr = opt_dict["tau_tr"]
        tau_grad = opt_dict["tau_gr"]
        step_size_ls = opt_dict["line search starting step size"]
        max_line_search_obj_closure_evals = opt_dict["maximum number of line search objective closure evaluations"]
        tau_ls = opt_dict["tau_ls"]
        stencil_wins = opt_dict["stencil_wins?"]
        num_dirs_custom = opt_dict["number of custom sampler directions"]
        beta_grad = opt_dict["stencil gradient beta"]
        
        obj = opt.step(scaled_obj_closure, 
                       stencil_type,
                       tau_tr,
                       tau_grad,
                       step_size_ls,
                       max_line_search_obj_closure_evals,
                       tau_ls,
                       stencil_wins,
                       custom_sampler,
                       num_dirs_custom,
                       beta_grad)
                       
    
    elif opt_name in ["SG-PGD"]:
        if sampler is None:
            custom_sampler_type = opt_dict["custom sampler"]
            
            if custom_sampler_type not in ["None"]:
                custom_sampler_dict = {"name":custom_sampler_type, 
                                       "dimension":sum(p.numel() for p in model.parameters())}
                custom_sampler = samplers.get_vector_sampler(custom_sampler_dict)
            
            else:
                custom_sampler = None
                
        stencil_type = opt_dict["stencil type"]
        h = opt_dict["h"]
        step_size_ls = opt_dict["line search starting step size"]
        max_line_search_obj_closure_evals = opt_dict["maximum number of line search objective closure evaluations"]
        tau_ls = opt_dict["tau_ls"]
        num_dirs_custom = opt_dict["number of custom sampler directions"]
        beta_grad = opt_dict["stencil gradient beta"]
        c = opt_dict["Armijo condition c"]
        epsilon_a = opt_dict["epsilon_a"]
        
        obj = opt.step(scaled_obj_closure,
                       stencil_type,
                       h,
                       step_size_ls,
                       max_line_search_obj_closure_evals,
                       tau_ls,
                       custom_sampler,
                       num_dirs_custom,
                       beta_grad,
                       c,
                       epsilon_a)
                       
    
    elif opt_name in ["NNAIF"]:
        if sampler is None:
            custom_sampler_type = opt_dict["custom sampler"]
            
            if custom_sampler_type not in ["None"]:
                custom_sampler_dict = {"name":custom_sampler_type, 
                                       "dimension":sum(p.numel() for p in model.parameters())}
                custom_sampler = samplers.get_vector_sampler(custom_sampler_dict)
            
            else:
                custom_sampler = None
                
        stencil_type = opt_dict["stencil type"]
        tau_tr = opt_dict["tau_tr"]
        tau_grad = opt_dict["tau_gr"]
        h_surr_min = opt_dict["h_min^surr"]
        eps_decrease = opt_dict["eps_dec^surr"]
        step_size_ls = opt_dict["line search starting step size"]
        max_line_search_obj_closure_evals = opt_dict["maximum number of line search objective closure evaluations"]
        tau_ls = opt_dict["tau_ls"]
        stencil_wins = opt_dict["stencil_wins?"]
        num_dirs_custom = opt_dict["number of custom sampler directions"]
        beta_grad = opt_dict["stencil gradient beta"]
        output_transform_dict = opt_dict["output transform dictionary"]
        output_transform = get_surrogate_model_output_transform(output_transform_dict)
        surrogate_fit_opt_dict = opt_dict["surrogate fit optimizer dictionary"]
        surrogate_descent_opt_dict = opt_dict["surrogate descent optimizer dictionary"]
        num_dirs_surrogate = opt_dict["number of points desired to pass surrogate filter test"]
        max_surrogate_filter_iter = opt_dict["maximum number of surrogate filtered sampling iterations"]
        
        obj = opt.step(scaled_obj_closure,
                       output_transform,
                       surrogate_fit_opt_dict,
                       surrogate_descent_opt_dict,
                       stencil_type,
                       tau_tr,
                       tau_grad,
                       h_surr_min,
                       eps_decrease,
                       step_size_ls,
                       max_line_search_obj_closure_evals,
                       tau_ls,
                       stencil_wins,
                       custom_sampler,
                       num_dirs_custom,
                       beta_grad,
                       num_dirs_surrogate,
                       max_surrogate_filter_iter)
    
    
    else:
        raise ValueError("The optimizer %s is not implemented..." % opt_name)
    
    
    return obj
    

def check_for_bad_values(opt_dict,
                         opt,
                         verbose=False):
    """
    Check if important optimizer values are inappropriate (e.g. NaN or Inf) so that corrective action can be taken.
    
    Args:
        opt_dict (dict): dictionary storing optimizer information.
        opt (torch.optim.Optimizer): PyTorch optimizer.
        verbose (bool): print additional information, useful for debugging (default: False).
        
    Returns:
        bad_values (bool): True if any important optimizer values are inappropriate, False otherwise.
    """
    opt_name = opt_dict["name"]
    opt_state = opt.state[opt._params[0]]
    
    
    # Custom optimizers
    # ===============
    if opt_name in ["CMA-ES"]:
        prev_obj = opt_state.get("previous objective value")
        check_val_list = [prev_obj]


    elif opt_name in ["EMNA"]:
        prev_obj_min = opt_state.get("previous objective value")
        check_val_list = [prev_obj_min]
        
    
    elif opt_name in ["IMFIL"]:
        prev_obj = opt_state.get("previous best objective value")
        prev_flat_stencil_grad = opt_state.get("previous flat implicit filtering stencil gradient")
        prev_flat_stencil_grad_norm = torch.linalg.norm(prev_flat_stencil_grad)
        check_val_list = [prev_obj, prev_flat_stencil_grad_norm.item()]
        
        # Stop if stencil size h is too small
        h = opt_state.get("h")
        h_min = opt_dict["h_min"]
        if h <= h_min:
            check_val_list.append(torch.nan)
            
        
    elif opt_name in ["SG-PGD"]:
        prev_obj = opt_state.get("previous best objective value")
        prev_flat_stencil_grad = opt_state.get("previous flat stencil gradient")
        prev_flat_stencil_grad_norm = torch.linalg.norm(prev_flat_stencil_grad)
        check_val_list = [prev_obj, prev_flat_stencil_grad_norm.item()]
        
        
    elif opt_name in ["NNAIF"]:
        prev_obj = opt_state.get("previous best objective value")
        prev_flat_stencil_grad_IF = opt_state.get("previous flat implicit filtering stencil gradient")
        prev_flat_stencil_grad_IF_norm = torch.linalg.norm(prev_flat_stencil_grad_IF)
        fs = opt_state.get("fs")
        check_val_list = fs.tolist()
        check_val_list.append(prev_obj)
        check_val_list.append(prev_flat_stencil_grad_IF_norm.item())
        
        # Stop if stencil size h is too small
        h = opt_state.get("h")
        h_min = opt_dict["h_min"]
        if h <= h_min:
            check_val_list.append(torch.nan)
    
    
    # Perform the checks
    if verbose:
        print(check_val_list)
    
    condition_evaluations = [math.isnan(val) or math.isinf(val) for val in check_val_list]

    if sum(condition_evaluations) > 0:
        bad_values = True

    else:
        bad_values = False


    return bad_values
    

def run_experiments(experiment_dict, 
                    save_directory_base, 
                    reset, 
                    use_cuda, 
                    quiet=False,
                    dtype=torch.float):
    """
    Run experiments.
    
    Args:
        experiment_dict (dict): dictionary storing experiment information.
        save_directory_base (string): directory where experiment information and results will be saved.
        reset (bool): flag indicating whether to ignore any existing experiment data and start over.
        use_cuda (bool): flag indicating whether to use GPU.
        quiet (bool): do not print additional information (default: False).
        dtype (dtype or string): the desired data type (default: torch.float).
        
    Returns:
        score_list (list): list of dictionaries containing recorded measurements. 
    """
    # Bookkeeping
    # ===============

    # Get experiment directory
    exp_id = hu.hash_dict(experiment_dict)
    save_directory = os.path.join(save_directory_base, 
                                  exp_id)

    if reset:
        # Delete and backup previous experiment
        hc.delete_experiment(save_directory, 
                             backup_flag=True)
    
    # Create save directory and save the experiment dictionary
    os.makedirs(save_directory, 
                exist_ok=True)
    hu.save_json(os.path.join(save_directory, 'experiment_dictionary.json'), 
                 experiment_dict)    
    
    # Print the experiment dictionary and save directory
    if not quiet:
        pprint.pprint(experiment_dict)
        print('Experiment saved in %s' % save_directory)  

    # Set seed for generating random numbers
    # ===============
    seed = 42 + experiment_dict["runs"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device setup
    # ===============
    if use_cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'CUDA is not available, please run with "-c 0"'
    else:
        device = 'cpu'

    # Print device
    if not quiet:
        print('Running on device: %s' % device)


    # CUTEst problem setup
    # ===============
    problem_name = experiment_dict["CUTEst problem dictionary"]["CUTEst problem"]
    sif_params = experiment_dict["CUTEst problem dictionary"]["SIF params"]
    cutest_problem = pycutest.import_problem(problem_name, 
                                             sifParams=sif_params)
    
    if experiment_dict["noise type"] is None:
        model = models.get_cutest_problem(problem_name, 
                                          sif_params,
                                          dtype=dtype).to(device=device)
    
    elif experiment_dict["noise type"] in ["additive"]:
        # Set objective function noise
        obj_noise_dict = copy.deepcopy(experiment_dict["objective noise dictionary"])
        obj_noise_func = samplers.get_sampler(obj_noise_dict)
        
        # Set gradient noise
        grad_noise_dict = copy.deepcopy(experiment_dict["gradient noise dictionary"])
        grad_noise_dict["dimension"] = cutest_problem.n
        grad_noise_func = samplers.get_sampler(grad_noise_dict)
        
        model = models.get_cutest_problem_additive_noise(problem_name,
                                                         sif_params,
                                                         obj_noise_func,
                                                         grad_noise_func,
                                                         dtype=dtype).to(device=device)   

    elif experiment_dict["noise type"] in ["additive scaled by starting point"]:
    
        # Evaluate objective function and its gradient at starting point            
        f0, g0 = cutest_problem.obj(cutest_problem.x0, 
                                    gradient=True)
        
        # Set objective function noise
        obj_noise_dict = copy.deepcopy(experiment_dict["objective noise dictionary"])
        noise_radius = obj_noise_dict["scaling factor"]*abs(f0)
        obj_noise_dict["radius"] = noise_radius
        obj_noise_func = samplers.get_sampler(obj_noise_dict)
        
        if not quiet:
            print("Initial f (f0): {}".format(f0))
            print("Noise radius: {}".format(noise_radius))
            
        # Set gradient noise
        grad_noise_dict = copy.deepcopy(experiment_dict["gradient noise dictionary"])
        grad_noise_dict["dimension"] = cutest_problem.n
        grad_noise_dict["radius"] = grad_noise_dict["scaling factor"]*float(np.linalg.norm(g0))
        grad_noise_func = samplers.get_sampler(grad_noise_dict)

        model = models.get_cutest_problem_additive_noise(problem_name,
                                                         sif_params,
                                                         obj_noise_func,
                                                         grad_noise_func,
                                                         dtype=dtype).to(device=device)

    # Print CUTEst problem information                                                             
    if not quiet:
        print(pycutest.problem_properties(problem_name))
        pycutest.print_available_sif_params(problem_name)
        print("Noise type: {}".format(experiment_dict["noise type"]))


    # Optimizer setup
    # ===============
    opt_dict = experiment_dict["optimizer dictionary"]
    opt = optimizers.get_optimizer(opt_dict=opt_dict,
                                   model=model)
    num_samples = experiment_dict["number of samples"]


    # Sampler setup
    # ===============
    if opt_dict.get("sampler", None) is not None:
        sampler_type = opt_dict["sampler"]
        sampler_dict = {"name":sampler_type, 
                        "dimension":sum(p.numel() for p in model.parameters())}
        sampler = samplers.get_vector_sampler(sampler_dict)
    else:
        sampler = None

    
    # Checkpointing setup
    # ===============

    # Setup checkpointing paths
    score_list_path = os.path.join(save_directory, "score_list.pkl")
    model_path = os.path.join(save_directory, "model_state_dict.pt")
    opt_path = os.path.join(save_directory, "optimizer_state_dict.pt")

    if os.path.exists(score_list_path) and os.path.exists(model_path):
        # Resume experiment from checkpoint
        score_list = hu.load_pkl(score_list_path)
        model.load_state_dict(torch.load(model_path))
        opt.load_state_dict(torch.load(opt_path))
        start_iter = score_list[-1]["iteration count"] + 1
        
        # Load counters
        total_obj_closure_evals = score_list[-1]["Total Objective Closure Evaluations"]
        total_func_evals = score_list[-1]["Total Function Evaluations"]
        
        
    else:
        # Restart experiment
        score_list = []    
        start_iter = 0
        
        # Initialize counters
        total_obj_closure_evals = 0
        total_func_evals = 0
        
        # Initial values at optimization starting point
        score_dict = {}
        score_dict["Iteration #"] = start_iter
        score_dict["Iteration Time (s)"] = 0.0
        score_dict["Number Of Samples"] = num_samples
        score_dict["Objective Closure Evaluation Count"] = 0
        score_dict["Total Objective Closure Evaluations"] = total_obj_closure_evals
        score_dict["Function Evaluations"] = 0        
        score_dict["Total Function Evaluations"] = total_func_evals
        
        if experiment_dict["noise type"] is not None:
            alpha = opt_dict["alpha"]
            
            # Compute true objective function value at optimization starting point
            true_obj = model.get_true_obj(model.get_x())/alpha
            score_dict["True Objective Function Value"] = true_obj
            true_min = experiment_dict["CUTEst problem dictionary"]["solution"]/alpha
            true_opt_gap = true_obj - true_min
            score_dict["log10(True Optimality Gap)"] = torch.log10(torch.tensor(true_opt_gap)).item()       
            
            # Compute true gradient norm at optimization starting point 
            true_grad_tensor = model.get_true_grad(model.get_x())/alpha
            true_grad_norm = torch.linalg.norm(true_grad_tensor).item()
            score_dict["True Gradient Norm"] = true_grad_norm
            
        # Add initial score dict to score list
        score_list += [score_dict]

    # Optimization
    # ===============

    # Start optimization
    if not quiet:
        print('Starting experiment at iteration %d' % (start_iter))      

    while total_func_evals < experiment_dict["maximum number of function evaluations"]:   
        # Set seed
        seed = total_func_evals + experiment_dict["runs"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)        
            
        # Optimize CUTEst problem
        # ===============                
        model.train()
        if not quiet:
            print("Optimizing %s using %s ..." % (problem_name, opt_dict["name"]))    
        
        start_time = time.time()
        opt_step(opt_dict, 
                 opt,
                 model, 
                 num_samples,
                 sampler=sampler)                                
        end_time = time.time()     
        
        # Compute true values
        # ===============   
        score_dict = {}
        
        if experiment_dict["noise type"] is not None:
            alpha = opt_dict["alpha"]
            
            # Compute true objective function value
            true_obj = model.get_true_obj(model.get_x())/alpha
            score_dict["True Objective Function Value"] = true_obj
            true_min = experiment_dict["CUTEst problem dictionary"]["solution"]/alpha
            true_opt_gap = true_obj - true_min
            score_dict["log10(True Optimality Gap)"] = torch.log10(torch.tensor(true_opt_gap)).item()       
            
            # Compute true gradient norm
            true_grad_tensor = model.get_true_grad(model.get_x())/alpha
            true_grad_norm = torch.linalg.norm(true_grad_tensor).item()
            score_dict["True Gradient Norm"] = true_grad_norm
        
        # Record iteration information
        # ===============
        opt_state = opt.state[opt._params[0]]
        score_dict["Iteration #"] = opt_state.get("iteration count")
        score_dict["Iteration Time (s)"] = end_time - start_time          
        
        # Number of samples
        score_dict["Number Of Samples"] = num_samples
        
        # Record objective function closure evaluation counts 
        obj_closure_eval_count = opt_state.get("objective closure evaluation count")
        score_dict["Objective Closure Evaluation Count"] = obj_closure_eval_count        
        total_obj_closure_evals += obj_closure_eval_count
        score_dict["Total Objective Closure Evaluations"] = total_obj_closure_evals
        
        # Record function evaluation counts 
        func_evals = obj_closure_eval_count*num_samples
        score_dict["Function Evaluations"] = func_evals
        total_func_evals += func_evals
        score_dict["Total Function Evaluations"] = total_func_evals
        
        # Add score dict to score list
        score_list += [score_dict]
        
        # Report and save
        if not quiet:
            print(pd.DataFrame(score_list).tail())
            
        hu.save_pkl(score_list_path,
                    score_list)
        
        # End experiment early if values become inappropriate (e.g. NaN or Inf)
        bad_values = check_for_bad_values(opt_dict,
                                          opt)
        
        if bad_values:
            break
            
        else:
            hu.torch_save(model_path,
                          model.state_dict())
            hu.torch_save(opt_path,
                          opt.state_dict())
        
        if not quiet:
            print("Saved at: %s" % save_directory)
            
    if not quiet:
        print("Experiment completed.")
        
    if device in ['cuda']:
        with torch.no_grad():
            torch.cuda.empty_cache()
        
    return score_list
    

if __name__ == '__main__':
    # Parse command line arguments
    # ===============
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dimension_category', nargs='+', help='Low, Medium, High or Very High Dimensional')
    parser.add_argument('-b', '--benchmark', nargs='+', help='Benchmark name')
    parser.add_argument('-sb', '--save_directory_base', required=True, help='Base folder for saving results')
    parser.add_argument('-pc', '--pycutest_cache', required=True, help='Location of PyCUTEst cache')
    parser.add_argument('-r', '--reset', type=bool, default=False, help='Ignore previous results and start over')
    parser.add_argument('-ei', '--exp_id', default=None, help='Experiment ID')
    parser.add_argument('-c', '--use_cuda', type=bool, default=False, help='Use GPU?')
    parser.add_argument('-q', '--quiet', type=bool, default=False, help='Suppress some printed messages?')
    args = parser.parse_args()
    
    # Show parsed arguments
    if not args.quiet:
        print("Parsed arguments:")
        print(args)  
        
    # PyCUTEst cache setup
    sys.path.append(args.pycutest_cache)        

    # Collect experiments
    # ===============
    if args.exp_id is not None:
        # Select one experiment
        save_directory = os.path.join(args.save_directory_base, 
                                      args.exp_id)
        experiment_dict = hu.load_json(os.path.join(save_directory, 'experiment_dictionary.json'))
        exp_list = [experiment_dict]

    else:
        # Select experiment group
        exp_list = []
        
        if args.dimension_category in ["Low Dimensional", "low dimensional"]:
            exp_list += exp_configs.LOW_DIM_CUTEST_EXPS[args.benchmark]
        
        elif args.dimension_category in ["Medium Dimensional", "medium dimensional"]:
            exp_list += exp_configs.MED_DIM_CUTEST_EXPS[args.benchmark]
        
        elif args.dimension_category in ["High Dimensional", "high dimensional"]:
            exp_list += exp_configs.HIGH_DIM_CUTEST_EXPS[args.benchmark]
            
        elif args.dimension_category in ["Very High Dimensional", "very high dimensional"]:
            exp_list += exp_configs.VERY_HIGH_DIM_CUTEST_EXPS[args.benchmark]

        else:
            raise ValueError("The dimension category %s is not implemented..." % args.dimension_category)
        
    # Run experiments
    # ===============
    for experiment_dict in exp_list:
        run_experiments(experiment_dict=experiment_dict, 
                        save_directory_base=args.save_directory_base,
                        reset=args.reset,
                        use_cuda=args.use_cuda,
                        quiet=args.quiet)    
               
        
