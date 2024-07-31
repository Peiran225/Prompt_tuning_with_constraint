#!/bin/bash
#SBATCH --job-name=R-subj  # Specify a name for your job
#SBATCH --output=outputs/roberta_subj.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/roberta_subj.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=40G                  # Memory per node (4GB in this example)
#SBATCH --qos huge-long



#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng



cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0

LEARNING_RATE=1e-3
LEARNING_RATE_LM=1e-3
EPOCH=1
MODEL_PATH="FacebookAI/roberta-base" # "gpt2"
GAMMA=6e-4
TASK="subj"
PROMPT_GROUP="TRUE"
PROMPT="subj_0"
NUM_OF_INITIAL_TEXT=1
SEED=42
# BASELINE_ONLY=True



cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for LEARNING_RATE in 1.1e-3 1.2e-3 1.3e-3 9e-2 8e-2 7e-2 
 # GAMMA 9e-7 7e-7 6e-7 5e-7 4e-7 3e-7 2e-7 1e-7 1e-6 1e-6 3e-6 4e-6 5e-6 1e-5 5e-5 1e-4 9e-4 8e-4     
 # GAMMA 9e-7 5e-4 2e-4 (best) 4e-4 3e-4 1e-4 7e-4 work
 # SEED works 60(best) 50 65 80 85 95 100
do
    python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT # --particular_layer=$PARTICULAR_LAYER
done



# PROMPT="subj_2"
# PARTICULAR_LAYER=4

# python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
#                              --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK --num_of_initial_text=$NUM_OF_INITIAL_TEXT \
#                              --baseline_only=$BASELINE_ONLY --particular_layer=$PARTICULAR_LAYER --prompt=$PROMPT \
#                              --base_initial=$BASE_INITIAL