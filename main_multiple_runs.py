import os
import sys
import subprocess
import argparse

# Define the list of optimizers and test problems
optimizers = ['EDRBO', 'EmpiricalUCB', 'UCB', 'WassersteinUCB', 'SBOKDE', 'DRBOKDE', 'MMD', 'MMD_Minimax_Approx', 'Stable']

# Synthetic benchmarks (Benchmark/Test_Function.py)
synthetic_problems = ['Ackley', 'ThreeHumpCamel', 'Hartmann', 'Hartmann_complicated', 'Modified_Branin', 'Continuous_Vendor', 'SixHumpCamel', 'portfolio_optimization', 'portfolio_normal_optimization']

# Real-world benchmarks (Benchmark/Real_World.py, see README_real_world.md)
real_world_problems = [
    # data-driven setting (contexts and regret under the same distribution)
    'Wind_Commitment', 'Wake_Steering', 'Wake_Steering_2C', 'Robot_Push',
    'Plant_Leaf', 'Glucose_Bolus', 'Building_Heating', 'Crop_Management',
    # distribution-shift setting (contexts from P_nominal, regret under P_eval)
    'Wind_Commitment_Shift', 'Wake_Steering_Shift', 'Wake_Steering_2C_Shift', 'Robot_Push_Shift',
    'Plant_Leaf_Shift', 'Glucose_Bolus_Shift', 'Building_Heating_Shift', 'Crop_Management_Shift',
]
# Further real-world variants: two-context versions, and the pushing reward for which
# Wasserstein robustness changes the decision
real_world_extra_problems = [
    'Building_Heating_2C', 'Building_Heating_2C_Shift', 'Crop_Management_2C', 'Crop_Management_2C_Shift',
    'Glucose_Bolus_2Meals', 'Glucose_Bolus_2Meals_Shift',
    'Robot_Push_Success', 'Robot_Push_Success_Shift',
]

suites = {
    'synthetic': synthetic_problems,
    'real_world': real_world_problems,
    'real_world_extra': real_world_extra_problems,
    'all': synthetic_problems + real_world_problems + real_world_extra_problems,
}

# # Define other parameters
init_size = 5 # Initial number of points
running_rounds = 200 # Number of total points 
repeat = 10 # Number of times to repeat each experiment
start_seed = 100
device = 'cuda'
beta = 1.5

parser = argparse.ArgumentParser(description="Run main.py for every combination of optimizer and test problem.")
parser.add_argument('--suite', choices=list(suites), default='synthetic',
                    help="Benchmarks to run: 'synthetic' (default), 'real_world', 'real_world_extra' or 'all'")
parser.add_argument('--benchmarks', nargs='+', default=None,
                    help="Explicit list of benchmarks (overrides --suite)")
parser.add_argument('--optimizers', nargs='+', default=optimizers,
                    help="Optimizers to run (default: all)")
parser.add_argument('--device', choices=['cpu', 'cuda'], default=device)
args = parser.parse_args()

test_problems = args.benchmarks or suites[args.suite]
main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')

# Loop through each combination of optimizer and test problem
for optimizer in args.optimizers:
    for test_problem in test_problems:
        # Construct the command to run the main script with the appropriate arguments
        command = [
            sys.executable, main_script,
            '--optimizer', optimizer,
            '--benchmark', test_problem,
            '--init_size', str(init_size),
            '--running_rounds', str(running_rounds),
            '--repeat', str(repeat),
            '--start_seed', str(start_seed),
            '--device', args.device,
            '--beta', str(beta)
        ]
        
        try:
            # Execute the command
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running optimizer {optimizer} on test problem {test_problem}: {e}")
