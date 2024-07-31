#!/bin/bash
#SBATCH --job-name=B-sst5  # Specify a name for your job
#SBATCH --output=outputs/bert_sst5.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/bert_sst5.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=40G                  # Memory per node (4GB in this example)
#SBATCH --qos medium



# SBATCH --account cbcb-heng
# SBATCH --partition cbcb-heng

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0


LEARNING_RATE=0.0075 # 0.1， 0.01 work
LEARNING_RATE_LM=0.0075
EPOCH=2
MODEL_PATH="bert-base-uncased" # "gpt2"
GAMMA=5e-4 
TASK="sst-5"
NUM_OF_INITIAL_TEXT=10000
# BASELINE_ONLY=True
SEED=43
SIMILARITY="L2_LM_prompt_only"
PROMPT="sst-5_0"

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for SEED in 43 
# LEARNING_RATE, working: 0.008 0.0082 0.0084 0.0075(best)
# GAMMA: working 5e-4(best), 1e-4, 1e-5
# SEED working: 43(best) 44 46 47 49 
do
python3 main_mean_of_prompts_prompt_only.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT \
                             --base_initial=$BASE_INITIAL --similarity=$SIMILARITY
done


# PROMPT="sst-5_0"
# PARTICULAR_LAYER=-2 # 10

# python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
#                              --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
#                              --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
#                              --seed=$SEED --particular_layer=$PARTICULAR_LAYER --prompt=$PROMPT \
#                              # --base_initial=$BASE_INITIAL