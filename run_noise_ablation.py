#!/usr/bin/env python3
"""
Sensor Noise Ablation Study Wrapper
Inherits from complete_distributed_experiment to run targeted noise profiles.
"""

import os
from pathlib import Path
import time
from complete_distributed_experiment import ExperimentConfig, DistributedExperimentManager

def run_ablation_study():
    print("="*60)
    print("STARTING SENSOR NOISE ABLATION STUDY")
    print("="*60)
    
    # 1. Lock in the "Sweet Spot" parameters
    sweet_spot_grids = [(15,15), (20, 20), (25, 25), (30, 30), (40, 40), (45, 45), (50, 50)]
    sweet_spot_agents = [2, 3]
    sweet_spot_patterns = ['stationary', 'evasive']
    sweet_spot_intervals = [5, 10, 25]
    
    # We only need to compare the baseline, the worst, and your best method
    target_methods = ['arithmetic_mean', 'standard_kl', 'weighted_visits_kl']
    
    # 2. Define the Sensor Noise Profiles
    # alpha = false positive rate (ghosts), beta = false negative rate (misses)
    noise_profiles = [
        {"name": "01_High_Quality", "alpha": 0.05, "beta": 0.10},
        {"name": "02_Baseline",     "alpha": 0.10, "beta": 0.20},
        {"name": "03_Degraded",     "alpha": 0.20, "beta": 0.30},
        {"name": "04_Ghost_Heavy",  "alpha": 0.30, "beta": 0.10},
        {"name": "05_perfect_sensing",   "alpha": 0.0, "beta": 0.0}
    ]
    
    overall_start = time.time()
    
    # 3. Loop through profiles and run the distributed manager for each
    for profile in noise_profiles:
        profile_name = profile["name"]
        print(f"\n\n{'='*40}")
        print(f"LAUNCHING PROFILE: {profile_name}")
        print(f"Alpha: {profile['alpha']} | Beta: {profile['beta']}")
        print(f"{'='*40}")
        
        # Create isolated directories so checkpoints never collide
        ckpt_dir = f"checkpoints_ablation/{profile_name}"
        res_dir = f"results_ablation/{profile_name}"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        
        # Instantiate the config from your original class
        config = ExperimentConfig(
            grid_sizes=sweet_spot_grids,
            n_agents_list=sweet_spot_agents,
            target_patterns=sweet_spot_patterns,
            merge_intervals=sweet_spot_intervals,
            merge_methods=target_methods,
            alpha=profile['alpha'],
            beta=profile['beta'],
            n_trials=30,     # 30 is plenty for statistical significance in an ablation
            max_steps=2500,  # Longer horizon to see noise effects
            fast_mode=False  # TRUE MPC
        )
        
        # Run your original parallel manager
        # It will automatically use all 48/100 cores on the Turing node for this specific profile
        manager = DistributedExperimentManager(
            config=config,
            checkpoint_dir=ckpt_dir,
            results_dir=res_dir
        )
        
        manager.run_distributed_experiment()
        print(f"Finished {profile_name}. Moving to next profile...")
        
    total_time = (time.time() - overall_start) / 3600
    print(f"\nAll noise profiles completed successfully in {total_time:.2f} hours!")

if __name__ == "__main__":
    run_ablation_study()