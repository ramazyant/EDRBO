To produce all the experiments, using Python 3.11.13, and having installed all requirements, run

> python main_multiple_runs.py

To run a separate experiment, run 

> python main.py --optimizer OPT --benchmark BENCH

where OPT is the optimisation algorithm from the following list

> LEDRBO, EmpiricalUCB, UCB, WassersteinUCB, SBOKDE, DRBOKDE, MMD, MMD_Minimax_Approx, Stable

and BENCH is the test function from the following list

> Ackley, ThreeHumpCamel, Hartmann, Hartmann_complicated, Modified_Branin, Continuous_Vendor, SixHumpCamel, portfolio_optimization, portfolio_normal_optimization

We include in the Results directory all obtained results.
