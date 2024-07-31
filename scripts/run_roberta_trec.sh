#!/bin/bash
#SBATCH --job-name=trec  # Specify a name for your job
#SBATCH --output=outputs/roberta_trec.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/roberta_trec.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=40G                  # Memory per node (4GB in this example)
#SBATCH --qos medium



#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng



cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0


LEARNING_RATE=3e-3 
LEARNING_RATE_LM=3e-3 
EPOCH=2
MODEL_PATH="FacebookAI/roberta-base" # "gpt2"
GAMMA=2e-4
TASK="trec"
PROMPT_GROUP="TRUE"
PROMPT="trec_0"
NUM_OF_INITIAL_TEXT=10000
# BASELINE_ONLY=True
SEED=42
# BASE_INITIAL="Random"

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for GAMMA in 2e-4
#LEARNING_RATE working: 3e-3(best) 4e-3 5e-3 not working: 6e-4 7e-4 8e-4 9e-4 1e-3 2e-3
#GAMMA not working: 7e-3  6e-3 5e-3 4e-3 3e-3 2e-3 9e-4 8e-4  6e-4 5e-4 4e-4   5e-5 1e-5
#GAMMA work: 1e-3 7e-4 3e-4 2e-4(best) 1e-4
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