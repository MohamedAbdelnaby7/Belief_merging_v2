#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=100
#SBATCH --mem=200G
#SBATCH -J "BeliefAblation"
#SBATCH -p long
#SBATCH -t 48:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

cd $SLURM_SUBMIT_DIR
module load python/3.10.17/v6xrl7k 2>/dev/null

export GRB_LICENSE_FILE=/home/$USER/gurobi.lic
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run the wrapper directly
python3 /home/mabdelnaby/belief_merging_NN/run_noise_ablation.py