#!/bin/bash
#SBATCH --job-name=B-subj  # Specify a name for your job
#SBATCH --output=outputs/bert_subj.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/bert_subj.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=40G                  # Memory per node (4GB in this example)
#SBATCH --qos medium



#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng



cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0

LEARNING_RATE=1e-3 # 0.1， 0.01 work
LEARNING_RATE_LM=1e-3
EPOCH=1
MODEL_PATH="bert-base-uncased" # "gpt2"
GAMMA=5e-5
TASK="subj"
NUM_OF_INITIAL_TEXT=10000
# BASELINE_ONLY=True
PROMPT="subj_4"
PROMPT_GROUP="TRUE"
SEED=50   # 42

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for GAMMA in 5e-7
#GAMMA 1 5e-1 1e-1 1e-3 5e-4 1e-4 5e-5 1e-5 1e-6  1e-7 5e-8 1e-8 1e-7 2e-7  4e-7 8e-7 does not work, 5e-7(best) 3e-7 6e-7 7e-7 9e-7 work
#seed 42 work
do
    python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT --prompt_group=$PROMPT_GROUP
done



# PROMPT="subj_2"
# PARTICULAR_LAYER=-2 # 4

# python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
#                              --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK --num_of_initial_text=$NUM_OF_INITIAL_TEXT \
#                              --baseline_only=$BASELINE_ONLY --particular_layer=$PARTICULAR_LAYER --prompt=$PROMPT \
#                              # --base_initial=$BASE_INITIAL