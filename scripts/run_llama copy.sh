#!/bin/bash
#SBATCH --job-name=constrained_prompting_job  # Specify a name for your job
#SBATCH --output=outputs_llama/prompt_tuning_with_layers.log       # Specify the output log file
#SBATCH --error=errors_llama.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:2     # originally rtxa6000
#SBATCH --mem=32gb                 # Memory per node (4GB in this example) 100? 

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

# Navigate to the directory containing your Python code


cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

# Activate your base environment (replace "your_base_env" with the name of your environment)
#conda init bash
# conda activate c-prompting

# Execute your Python code
python3 main_llama.py
# Deactivate the environment (if you want to)
# conda deactivate

# Your job is done!

