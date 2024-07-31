#!/bin/bash
#SBATCH --job-name=constrained_prompting_job  # Specify a name for your job
#SBATCH --output=output_constrained_prompting.log       # Specify the output log file
#SBATCH --error=error_constrained_prompting.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --time=01:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa4000:1     # to 2?
#SBATCH --mem=32G                  # Memory per node (4GB in this example) 100? 

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

# Navigate to the directory containing your Python code
# cd /fs/nexus-scratch/peiran/prompting_with_constraints

# Activate your base environment (replace "your_base_env" with the name of your environment)
#conda init bash
#conda activate prompting

# Execute your Python code
python3 test.py
# Deactivate the environment (if you want to)
# conda deactivate

# Your job is done!

