#!/bin/bash
#SBATCH --job-name=gpt2_small # Specify a name for your job
#SBATCH --output=outputs/outputs_gpt2_small/sst_5_different_prompts_outputs.log       # Specify the output log file
#SBATCH --error=errors/sst5_different_prompts_errors.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1     # to 2?
#SBATCH --mem=32gb                  # Memory per node (4GB in this example) 100? 
#SBATCH --qos huge-long
#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

# Navigate to the directory containing your Python code
cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

# Activate your base environment (replace "your_base_env" with the name of your environment)
#conda init bash
# conda activate c-prompting

# Execute your Python code
CUDA_VISIBLE_DEVICES=0

LEARNING_RATE=1e-2 # SST-2: 1e-3 
LEARNING_RATE_LM=1e-2
EPOCH=1
MODEL_PATH="gpt2" # "gpt2"
GAMMA=5e-5 # 1e-8 upgrade to 0.86 at layer 3 for gpt2 small, 5e-5 for bert
TASK="sst-5"

# for TASK in 'sst-5' # 'SST-2' 'sst-5' 'agnews' 'trec' 'subj'
# do
python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK 
# done

# Deactivate the environment (if you want to)
# conda deactivate

# Your job is done!

