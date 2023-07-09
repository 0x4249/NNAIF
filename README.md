# The Neural Network Accelerated Implicit Filtering (NNAIF) Paper
This Git repository contains the Python code for running the numerical experiments discussed in the paper
"Neural Network Accelerated Implicit Filtering: Integrating Neural Network Surrogates With Provably Convergent Derivative Free Optimization Methods" by Brian Irwin, Eldad Haber, Raviv Gal, and Avi Ziv. The ICML 2023 paper can be found on OpenReview at [https://openreview.net/forum?id=KG1eLtsX61](https://openreview.net/forum?id=KG1eLtsX61).


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

To run the low dimensional (0 < d <= 50) CUTEst experiments from Section 5.2 of the paper, using a GPU if available,
execute the following via terminal:
```
python run_cutest_experiments.py -d "Low Dimensional" -b "CUTEst Scaled Noise" -sb "<save-location>" 
-pc "<PyCUTEst-cache-location>" -c 1
```

To run the medium dimensional (50 < d <= 200) (0 < d <= 50) CUTEst experiments from Section 5.2 of the paper, using a
GPU if available, execute the following via terminal:
```
python run_cutest_experiments.py -d "Medium Dimensional" -b "CUTEst Scaled Noise" -sb "<save-location>" 
-pc "<PyCUTEst-cache-location>" -c 1
```

To run the high dimensional (200 < d <= 3000) CUTEst experiments from Section 5.2 of the paper, using a GPU if 
available, execute the following via terminal:
```
python run_cutest_experiments.py -d "High Dimensional" -b "CUTEst Scaled Noise" -sb "<save-location>" 
-pc "<PyCUTEst-cache-location>" -c 1
```

To run the very high dimensional (d > 3000) CUTEst experiments from Section 5.2 of the paper, using a GPU if available,
execute the following via terminal:
```
python run_cutest_experiments.py -d "High Dimensional" -b "CUTEst Scaled Noise" -sb "<save-location>" 
-pc "<PyCUTEst-cache-location>" -c 1
```

The experiment configurations can be changed by editing the files in the folder "exp_configs".

The numerical experiments code contained in this Git repository was originally tested using Python 3.8.15 on a computer 
with the Ubuntu 20.04 LTS operating system installed.

# Citation
If you use this code, please cite the paper:

Irwin, B., Haber, E., Gal, R., Ziv, A. Neural Network Accelerated Implicit Filtering: Integrating Neural Network Surrogates With Provably Convergent Derivative Free Optimization Methods. *Proceedings of the 40th International Conference on Machine Learning*, 2023.

BibTeX: 
```
@inproceedings{irwin-nnaif-2023,
    Author = {Brian Irwin and Eldad Haber and Raviv Gal and Avi Ziv},
    Title = {Neural Network Accelerated Implicit Filtering: Integrating Neural Network Surrogates With Provably Convergent Derivative Free Optimization Methods},
    Booktitle = {Proceedings of the 40th International Conference on Machine Learning},
    Year = {2023},
    Series = {Proceedings of Machine Learning Research},
    Address = {Honolulu, USA},
    Publisher = {PMLR}
}
```


