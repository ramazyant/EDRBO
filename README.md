Official repository of [Ensemble Distributionally Robust Bayesian Optimization with Continuous Context](https://arxiv.org/abs/2605.07565).

This code is based on repositories by [Micheli F. et al.](https://github.com/frmicheli/WDRBO/tree/main) and [X. Huang et al.](https://github.com/lamda-bbo/sbokde.)

>To produce all the experiments, use Python 3.11.13 and install all requirements from ```requirements.txt```. Then run
```
python main_multiple_runs.py
```

>To run a separate experiment 
```
python main.py --optimizer OPT --benchmark BENCH
```
where OPT is the optimisation algorithm from the following list
```
LEDRBO, EmpiricalUCB, UCB, WassersteinUCB, SBOKDE, DRBOKDE, MMD, MMD_Minimax_Approx, Stable
```
and BENCH is the test function from the following list
```
Ackley, ThreeHumpCamel, Hartmann, Hartmann_complicated, Modified_Branin, Continuous_Vendor, SixHumpCamel, portfolio_optimization, portfolio_normal_optimization
```
We include all presented results in the Results directory.
