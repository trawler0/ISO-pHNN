#!/bin/bash
#SBATCH --job-name=si
#SBATCH --partition=normal
#SBATCH --account=imacm
#SBATCH -N 1                      # Number of nodes
#SBATCH --ntasks=1                # Total number of tasks (i.e., parallel jobs).
#SBATCH --cpus-per-task=2         # Number of CPUs per task.
#SBATCH --mem-per-cpu=4096 # in MB
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/%x_%j.out    # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.err

NAME=$1
SEED=$2
DATASET=$3
NUM_TRAJECTORIES=$4
EPOCHS=$5
HIDDEN_DIM=$6
LR=$7
J=$8
R=$9
G=${10}
GRAD_H=${11}
EXPERIMENT=${12}
RUN_NAME=${13}

echo "Args:"
echo "  NAME  : $NAME"
echo "  SEED  : $SEED"
echo "  DATASET  : $DATASET"
echo "  NUM_TRAJECTORIES  : $NUM_TRAJECTORIES"
echo "  EPOCHS  : $EPOCHS"
echo "  HIDDEN_DIM  : $HIDDEN_DIM"
echo "  LR  : $LR"
echo "  J  : $J"
echo "  R  : $R"
echo "  G  : $G"
echo "  GRAD_H  : $GRAD_H"
echo "  EXPERIMENT  : $EXPERIMENT"
echo "  RUN_NAME  : $RUN_NAME"
echo "baseline"

# Append NAME to the experiment flag.
experiment="${experiment}_${NAME}"

# --- Module Management ---
# Unload Anaconda3 to prevent conflicts.
module unload Anaconda3/2022.05

# Load Necessary Modules.
module load 2022a GCCcore/11.3.0 Python/3.10.4

pip install nonlinear_benchmarks

# --- Run Parallel Jobs ---
PYTHONUNBUFFERED=1 stdbuf -oL -eL srun --exclusive -n1 python -u main.py --name $NAME --seed $SEED --data $DATASET --num_trajectories $NUM_TRAJECTORIES --epochs $EPOCHS --hidden_dim $HIDDEN_DIM --lr $LR --J $J --R $R --G $G --grad_H $GRAD_H --experiment $EXPERIMENT --baseline --run_name $RUN_NAME --no-forecast | stdbuf -oL awk 'NR==1{print;next}{if ((systime()-t)>30){print; t=systime()}}'
