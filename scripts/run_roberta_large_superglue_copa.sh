#!/bin/bash
#SBATCH --job-name=copa_roberta_large  # Specify a name for your job
#SBATCH --output=outputs/copa_roberta_large.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/copa_roberta_large.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --mem=40G                  # Memory per node (4GB in this example)
#SBATCH --qos medium



#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng

cd /fs/nexus-scratch/peiran/Prompt_tuning_with_constraint

CUDA_VISIBLE_DEVICES=0


LEARNING_RATE=5e-4
LEARNING_RATE_LM=3e-3 
EPOCH=3
MODEL_PATH="FacebookAI/xlm-roberta-large" # "gpt2"
GAMMA=2e-4
TASK="copa"
PROMPT_GROUP="TRUE"
NUM_OF_INITIAL_TEXT=1 
PROMPT="copa_0"
# NUM_OF_INITIAL_TEXT=10000
# BASELINE_ONLY=True
SEED=42


for GAMMA in 1e-7 1e-8 1e-9 1e-10
#LEARNING_RATE working: 5e-4 not working: 1e-2 5e-3 1e-3  1e-4 5e-5 1e-5 5e-6 1e-6 
#GAMMA not working: 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6
#GAMMA work: 
do
    python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT # --particular_layer=$PARTICULAR_LAYER
done

# BASELINE_ONLY=False
# PROMPT="trec_1"
# PARTICULAR_LAYER=4
# python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
#                              --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
#                              --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
#                              --seed=$SEED --particular_layer=$PARTICULAR_LAYER --prompt=$PROMPT