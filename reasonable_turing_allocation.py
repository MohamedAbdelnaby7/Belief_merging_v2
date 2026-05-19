#!/usr/bin/env python3
"""
Reasonable Resource Allocation for Shared University Cluster - Optimized for Single-Node
"""

import json
import os
from pathlib import Path


class ReasonableResourceManager:
    """Conservative resource management for shared university clusters with multiple configurations"""
    
    def __init__(self):
        self.reasonable_limits = {
            'max_cores_per_job': 48, # Reasonable for university cluster        
            'max_memory_gb': 350,    # Conservative memory request        
            'max_time_hours': 150,   # max to get scheduled         
            'nodes': 1, # James's recommendation: Stick to one node
        }
    
    def calculate_reasonable_resources(self, experiment_type: str = "standard") -> dict:
        """Calculate single-node resource allocation to ensure tasks see all CPUs"""
        resource_tiers = {
            'test': {
                'cpus_per_task': 8,
                'memory_gb': 32,
                'time_hours': 2,
                'partition': 'short',
                'description': 'Quick testing'
            },
            'small': {
                'cpus_per_task': 32,
                'memory_gb': 64,
                'time_hours': 12,
                'partition': 'short',
                'description': 'Small scale experiment'
            },
            'standard': {
                'cpus_per_task': 48,
                'memory_gb': 200,
                'time_hours': 24,
                'partition': 'long',
                'description': 'Standard full experiment'
            },
            'large': {
                'cpus_per_task': 100, # maximum number of workers through personal gurobi license
                'memory_gb': 200,
                'time_hours': 120,
                'partition': 'long',
                'description': 'Large comprehensive study'
            }
        }
        return resource_tiers.get(experiment_type, resource_tiers['standard'])
    
    def create_reasonable_config(self, experiment_type: str = "standard") -> dict:
        """Create experiment configuration with proper MPC (no fast mode) for multiple grid sizes"""
        
        # Configuration based on available compute time
        config_templates = {
            'test': {
                'grid_sizes': [[10, 10], [20, 20]],  # Small grids for testing
                'n_agents_list': [2, 3],  # Fewer agents for quick test
                'n_trials': 10,
                'horizon': 2,
                'max_steps': 200,
                'merge_intervals': [5, 10, 50],
                'target_patterns': ['random'],
                'fast_mode': False,
                'random_walk_mode': False,
                #'merge_methods': ['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean']
                'merge_methods': ['standard_kl', 'reverse_kl']
            },
            'small': {
                'grid_sizes': [[10, 10], [20, 20]],
                'n_agents_list': [2, 3, 4],
                'n_trials': 20,
                'horizon': 2,
                'max_steps': 500,
                'merge_intervals': [0, 25, 100, float('inf')],
                'target_patterns': ['random', 'evasive'],
                'fast_mode': False,
                'random_walk_mode': False,
                'merge_methods': ['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean']
            },
            'standard': {
                'grid_sizes': [[10, 10], [20, 20], [30, 30]],
                'n_agents_list': [2, 3, 4],
                'n_trials': 50,
                'horizon': 3,
                'max_steps': 1000,
                'merge_intervals': [0, 10, 25, 50, 100, 200, float('inf')],
                'target_patterns': ['stationary', 'random', 'evasive', 'patrol'],
                'fast_mode': False,
                'random_walk_mode': False,
                'merge_methods': ['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean']
            },
            'large': {
                'grid_sizes': [[10, 10], [15, 15], [20, 20], [25, 25], [30, 30], [40, 40], [45, 45], [50, 50], [100, 100]],
                'n_agents_list': [2, 3],
                'n_trials': 10,
                'horizon': 3,
                'max_steps': 2500,
                'merge_intervals': [0, 5, 10, 25, 50, 100, 200, 500, float('inf')],
                'target_patterns': ['stationary', 'random', 'evasive', 'patrol'],
                'fast_mode': False,
                'random_walk_mode': False,
                'merge_methods': ['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean', 'weighted_visits_kl']
            }
        }
        
        base_config = config_templates.get(experiment_type, config_templates['standard'])
        
        # Add common parameters
        full_config = {
            'alpha': 0.1,  # False positive rate
            'beta': 0.2,   # False negative rate
            **base_config
        }
        
        return full_config
    
    def create_slurm_script(self, experiment_type: str = "standard") -> str:
        """Create reasonable SLURM script that will get scheduled"""
        
        res = self.calculate_reasonable_resources(experiment_type)
        
        script_content = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task={res['cpus_per_task']}
#SBATCH --mem={res['memory_gb']}G
#SBATCH -J "BeliefMerging-{experiment_type}"
#SBATCH -p {res['partition']}
#SBATCH -t {res['time_hours']:02d}:00:00
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@wpi.edu

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | CPUs: {res['cpus_per_task']}"

cd $SLURM_SUBMIT_DIR
module load python/3.10.17/v6xrl7k 2>/dev/null

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 complete_distributed_experiment.py \\
    --config-file configs/{experiment_type}_config.json \\
    --max-workers {res['cpus_per_task']} \\
    --checkpoint-dir checkpoints \\
    --results-dir results
"""
        return script_content

    def estimate_runtime(self, experiment_type: str = "standard") -> dict:
        config = self.create_reasonable_config(experiment_type)
        res = self.calculate_reasonable_resources(experiment_type)
        total_tasks = len(config['grid_sizes']) * len(config['n_agents_list']) * len(config['merge_intervals']) * len(config['target_patterns']) * config['n_trials'] * len(config['merge_methods'])
        est_parallel = (total_tasks * 20) / res['cpus_per_task'] / 3600
        return {'total_tasks': total_tasks, 'parallel_hours': est_parallel, 'will_fit_in_time_limit': est_parallel < res['time_hours']}

    def generate_all_configurations(self):
        Path("configs").mkdir(exist_ok=True)
        Path("scripts").mkdir(exist_ok=True)
        for exp_type in ['test', 'small', 'standard', 'large']:
            config = self.create_reasonable_config(exp_type)
            with open(Path("configs") / f"{exp_type}_config.json", 'w') as f: json.dump(config, f, indent=2)
            with open(Path("scripts") / f"run_{exp_type}.sh", 'w') as f: f.write(self.create_slurm_script(exp_type))
            os.chmod(Path("scripts") / f"run_{exp_type}.sh", 0o755)

if __name__ == "__main__":
    manager = ReasonableResourceManager()
    manager.generate_all_configurations()