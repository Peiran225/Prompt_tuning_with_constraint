#!/bin/bash
#SBATCH --job-name=sst5  # Specify a name for your job
#SBATCH --output=outputs/gpt2_large_sst5.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/gpt2_large_sst5.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=32G                  # Memory per node (4GB in this example)
#SBATCH --qos medium

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0


LEARNING_RATE=9e-3 # 5e-3 6e-3 7e-3 8e-3 9e-3 work
LEARNING_RATE_LM=9e-3
EPOCH=1
MODEL_PATH="gpt2-large"
GAMMA=8e-8
TASK="sst-5"
NUM_OF_INITIAL_TEXT=10000
# BASELINE_ONLY=True
PROMPT="sst-5_0"
SEED=42
PROMPT_GROUP="TRUE"
cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for SEED in 200
# GAMMA not working: 1e-7  2e-7 3e-7 4e-7 5e-7  9e-8 8e-8 7e-8 6e-8 5e-8 7.6e-8 7.7e-8 7.8e-8 7.9e-8 8.1e-8 8.2e-8 8.3e-8 8.4e-8 8.5e-8
# GAMMA working:  8e-8(best)  5e-7 7.5e-8
# seed: working: 50 60 65 70 80 85 95 200(best)
do
    python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT 
done



