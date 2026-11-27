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
EDRBO, EmpiricalUCB, UCB, WassersteinUCB, SBOKDE, DRBOKDE, MMD, MMD_Minimax_Approx, Stable
```
and BENCH is the test function from the following list
```
Ackley, ThreeHumpCamel, Hartmann, Hartmann_complicated, Modified_Branin, Continuous_Vendor, SixHumpCamel, portfolio_optimization, portfolio_normal_optimization
```

>Real-world benchmarks

`Benchmark/Real_World.py` adds 24 real-world benchmarks with continuous contexts: wind-power commitment, wind-farm wake steering (FLORIS), robot pushing (Box2D), plant growth, insulin dosing (UVA/Padova model), building heating (BOPTEST) and crop management (WOFOST). Each comes in a data-driven version and a distribution-shift version (`_Shift`: contexts observed from one distribution, regret measured under another). They have the same interface as the synthetic benchmarks and run in the same way (the packages they need are in ```requirements.txt```):
```
python main.py --optimizer EDRBO --benchmark Wake_Steering_Shift
python main_multiple_runs.py --suite real_world
```
BENCH can also be one of
```
Wind_Commitment, Wind_Commitment_Shift, Wake_Steering, Wake_Steering_Shift, Wake_Steering_2C, Wake_Steering_2C_Shift, Robot_Push, Robot_Push_Shift, Robot_Push_Success, Robot_Push_Success_Shift, Plant_Leaf, Plant_Leaf_Shift, Glucose_Bolus, Glucose_Bolus_Shift, Glucose_Bolus_2Meals, Glucose_Bolus_2Meals_Shift, Building_Heating, Building_Heating_Shift, Building_Heating_2C, Building_Heating_2C_Shift, Crop_Management, Crop_Management_Shift, Crop_Management_2C, Crop_Management_2C_Shift
```
See [Benchmark/README_real_world.md](Benchmark/README_real_world.md) for the problems, their data sources, the benchmark suites of `main_multiple_runs.py` and the worst-case evaluation of finished runs.
