#!/bin/bash
#SBATCH --job-name=agnews  # Specify a name for your job
#SBATCH --output=outputs/agnews_gpt2_small.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/agnews_gpt2_errors.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=2:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=32G                  # Memory per node (4GB in this example)
#SBATCH --qos medium

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0

LEARNING_RATE=1e-3 # 1e-3 work
LEARNING_RATE_LM=1e-3
EPOCH=1
MODEL_PATH="gpt2"
GAMMA=1e-8 # 1e-8 upgrade to 0.86 at layer 3 for gpt2 small, 5e-5 for bert
TASK="agnews"
PROMPT="agnews_0"
NUM_OF_INITIAL_TEXT=1
SEED=42

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for GAMMA in 1e-6 5e-7 1e-7 5e-8 1e-8 5e-9 1e-9 5e-10 1e-10
do
python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT \
                             --seed=$SEED  --prompt=$PROMPT 
done

# for LEARNING_RATE_LM in 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6
# do
#     python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
#                              --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK --num_of_initial_text=$num_of_initial_text
# done

