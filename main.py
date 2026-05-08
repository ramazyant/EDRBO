from edrbo.optimizers import UCB_Optimizer
from edrbo.optimizers import Stable_Optimizer
from edrbo.optimizers import MMD_Optimizer
from edrbo.optimizers import MMD_Minimax_Approx_Optimizer
from edrbo.optimizers import SBOKDE_Optimizer
from edrbo.optimizers import WassersteinUCB_Optimizer
from edrbo.optimizers import EmpiricalUCB_Optimizer
from edrbo.optimizers import DRBOKDE_Optimizer
from edrbo.optimizers import WDRBO3_Optimizer

from Benchmark.Test_Function import Ackley
from Benchmark.Test_Function import Hartmann
from Benchmark.Test_Function import Modified_Branin
from Benchmark.Test_Function import portfolio_optimization
from Benchmark.Test_Function import portfolio_normal_optimization
from Benchmark.Test_Function import Hartmann_complicated
from Benchmark.Test_Function import Continuous_Vendor
from Benchmark.Test_Function import ThreeHumpCamel
from Benchmark.Test_Function import SixHumpCamel

import argparse
import botorch
import numpy as np
import random
import torch
import pickle as pkl
import time
from multiprocessing import Pool, cpu_count
import warnings
import os


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

OPTIMIZERS = [
    "SBOKDE",
    "UCB",
    "Stable",
    "MMD",
    "MMD_Minimax_Approx",
    "WassersteinUCB",
    "EmpiricalUCB",
    "DRBOKDE",
    "WDRBO3",
]

PROBLEMS = [
    "ThreeHumpCamel",
    "Ackley",
    "Hartmann",
    "Hartmann_complicated",
    "Modified_Branin",
    "portfolio_optimization",
    "portfolio_normal_optimization",
    "Continuous_Vendor",
    "SixHumpCamel",
    "Ackley_dimensional"
]

parser.add_argument(
    '--optimizer',
    type=str,
    choices=OPTIMIZERS,
    default='SBOKDE',
    help="Optimization algorithm"
)
parser.add_argument(
    "--benchmark",
    type=str,
    choices=PROBLEMS,
    default="Ackley",
    help="Bechmark to test optimization algorithm on"
)
parser.add_argument(
    "--minimization",
    action="store_true",
    default=False,
    help="If specified then optimization problem will be solved as minimization problem"
)
parser.add_argument(
    "--init_size",
    type=int,
    default=5,
    help="Initial number of samples"
)
parser.add_argument(
    "--running_rounds",
    type=int,
    default=200,
)
parser.add_argument(
    "--repeat",
    type=int,
    default=10,
    help="Number of experiment repeats"
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=100,
    help="Starting seed"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default="cuda"
)
parser.add_argument(
    "--beta",
    type=float,
    default=1.5
)
parser.add_argument(
    "--dist_shift",
    type=bool,
    default=False
)

args = parser.parse_args()

if args.optimizer in ["MMD", "DRBOKDE"] and args.device == "cuda":
    print(f"GPU for {args.optimizer} is not supported. Set to cpu.")
    args.device = 'cpu'

print(args)
device = torch.device(args.device)

# Create the Result directory if it doesn't exist
os.makedirs('./Result', exist_ok=True)

test_func = eval(args.benchmark)(negate=not args.minimization).to(dtype=torch.float64)#

for i in range(args.repeat):
    
    start = time.time()
    seed = args.start_seed + i
    base_dir = f'{args.optimizer}_{args.benchmark}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_results'
    # check if the same experiment has been already done 
    if os.path.exists(f'./Result/{base_dir}.pkl'):
        print(f"Experiment {base_dir} has been already done.")
        continue
    
    # Set seed
    botorch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Optimization process
    opt = eval(args.optimizer+'_Optimizer')(test_func, running_rounds=args.running_rounds, init_size=args.init_size, device=device, beta=args.beta)
    X, Y, contexts, cumulative_stochastic_regret, cumulative_regret, best_Y, stochastic_Y = opt.run_opt()
    
    print(f"Time for {i}th run:{time.time() - start}")

    # Save results
    results = {
        'X': X.cpu().detach().numpy(),
        'Y': Y.cpu().detach().numpy(),
        'contexts': contexts.cpu().detach().numpy(),
        'stochastic_Y': stochastic_Y.cpu().detach().numpy(),
        'cumulative_regret': cumulative_regret,
        'best_Y': best_Y.cpu().detach().numpy(),
        'time': time.time() - start
    }

    with open(f'./Result/{base_dir}.pkl','wb') as f:
        pkl.dump(results, f)
    
    torch.cuda.empty_cache()


if __name__ == '__main__':
    seeds = [args.start_seed + i for i in range(args.repeat)]
    print("All done")
