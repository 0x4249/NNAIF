# The Neural Network Accelerated Implicit Filtering (NNAIF) Paper
This Git repository contains the Python code for running the numerical experiments discussed in the paper
"Neural Network Accelerated Implicit Filtering: Integrating Neural Network Surrogates With Provably Convergent Derivative Free Optimization Methods" by Brian Irwin, Eldad Haber, Raviv Gal, and Avi Ziv. The ICML 2023 paper is available at [https://proceedings.mlr.press/v202/irwin23a.html](https://proceedings.mlr.press/v202/irwin23a.html).


# Running The Code
## Section 5.1: Visualizing NNAIF Surrogate Models
To run the NNAIF surrogate model visualization experiment from Section 5.1 of the paper, run the Jupyter notebook:
```
visualize_NNAIF_surrogate_evolution.ipynb
```

## Section 5.2: CUTEst Problems With Additive Uniform Noise
For a description of the relevant arguments, execute the following via terminal:
```
python run_cutest_experiments.py --help
```
In what follows below, `<save-location>` is a placeholder for the path of the folder where results will be saved, and 
`<PyCUTEst-cache-location>` is a placeholder for the path of the PyCUTEst cache.

To run the low dimensional (0 < d <= 50) CUTEst experiments from Section 5.2 of the paper, using a GPU if available,
execute the following via terminal:
```
python run_cutest_experiments.py -d "Low Dimensional" -b "CUTEst Scaled Noise" -sb "<save-location>" -pc "<PyCUTEst-cache-location>" -c 1
```

To run the medium dimensional (50 < d <= 200) CUTEst experiments from Section 5.2 of the paper, using a
GPU if available, execute the following via terminal:
```
python run_cutest_experiments.py -d "Medium Dimensional" -b "CUTEst Scaled Noise" -sb "<save-location>" -pc "<PyCUTEst-cache-location>" -c 1
```

To run the high dimensional (200 < d <= 3000) CUTEst experiments from Section 5.2 of the paper, using a GPU if 
available, execute the following via terminal:
```
python run_cutest_experiments.py -d "High Dimensional" -b "CUTEst Scaled Noise" -sb "<save-location>" -pc "<PyCUTEst-cache-location>" -c 1
```

To run the very high dimensional (d > 3000) CUTEst experiments from Section 5.2 of the paper, using a GPU if available,
execute the following via terminal:
```
python run_cutest_experiments.py -d "Very High Dimensional" -b "CUTEst Scaled Noise" -sb "<save-location>" -pc "<PyCUTEst-cache-location>" -c 1
```

The CUTEst experiment configurations can be changed by editing the files in the folder "exp_configs". An example Jupyter notebook
for visualizing the results of the CUTEst numerical experiments is provided by:
```
visualize_CUTEst_results.ipynb
```

The numerical experiments code contained in this Git repository was originally tested using Python 3.8.15 on a desktop computer equipped with an NVIDIA RTX 2080 TI GPU and the Ubuntu 20.04 LTS Linux operating system. The file `environment.yml` describes the conda environment used to perform the numerical experiments.

# Citation
If you use this code, please cite the paper:

Irwin, B., Haber, E., Gal, R. & Ziv, A.. (2023). Neural Network Accelerated Implicit Filtering: Integrating Neural Network Surrogates With Provably Convergent Derivative Free Optimization Methods. *Proceedings of the 40th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 202:14376-14389 Available from https://proceedings.mlr.press/v202/irwin23a.html.

BibTeX: 
```
@InProceedings{pmlr-v202-irwin23a,
  title = 	 {Neural Network Accelerated Implicit Filtering: Integrating Neural Network Surrogates With Provably Convergent Derivative Free Optimization Methods},
  author =       {Irwin, Brian and Haber, Eldad and Gal, Raviv and Ziv, Avi},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {14376--14389},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/irwin23a/irwin23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/irwin23a.html},
  abstract = 	 {In this paper, we introduce neural network accelerated implicit filtering (NNAIF), a novel family of methods for solving noisy derivative free (i.e. black box, zeroth order) optimization problems. NNAIF intelligently combines the established literature on implicit filtering (IF) optimization methods with a neural network (NN) surrogate model of the objective function, resulting in accelerated derivative free methods for unconstrained optimization problems. The NN surrogate model consists of a fixed number of parameters, which can be as few as $\approx 1.3 \times 10^{4}$, that are updated as NNAIF progresses. We show that NNAIF directly inherits the convergence properties of IF optimization methods, and thus NNAIF is guaranteed to converge towards a critical point of the objective function under appropriate assumptions. Numerical experiments with $31$ noisy problems from the CUTEst optimization benchmark set demonstrate the benefits and costs associated with NNAIF. These benefits include NNAIF’s ability to minimize structured functions of several thousand variables much more rapidly than well-known alternatives, such as Covariance Matrix Adaptation Evolution Strategy (CMA-ES) and finite difference based variants of gradient descent (GD) and BFGS, as well as its namesake IF.}
}
```

