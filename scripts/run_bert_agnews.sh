#!/bin/bash
#SBATCH --job-name=agnews  # Specify a name for your job
#SBATCH --output=outputs/bert_agnews.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/bert_agnews.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=36:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --mem=40G                  # Memory per node (4GB in this example)
#SBATCH --qos huge-long



#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng


cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0


LEARNING_RATE=0.001
LEARNING_RATE_LM=0.001
EPOCH=1
MODEL_PATH="bert-base-uncased" # "gpt2"
GAMMA=1e-4
TASK="agnews"
NUM_OF_INITIAL_TEXT=1
# BASELINE_ONLY=False
SEED=42

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for SEED in  43 44 45 46 47 48 49 
#GAMMA: not working: 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 1 5e-1 1e-1 5e-2 1e-2, 0.0005 0.0001 not decrease but not better
do
    python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED
done