#!/bin/bash
#SBATCH --job-name=B-trec  # Specify a name for your job
#SBATCH --output=outputs/bert_trec.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/bert_trec.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=40G                  # Memory per node (4GB in this example)
#SBATCH --qos medium



#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng



cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0


LEARNING_RATE=0.052 # 0.1， 0.01 work
LEARNING_RATE_LM=0.052
EPOCH=2
MODEL_PATH="bert-base-uncased" # "gpt2"
GAMMA=7e-3
TASK="trec"
NUM_OF_INITIAL_TEXT=1
# BASELINE_ONLY=True
SEED=42
PROMPT="trec_1"
SIMILARITY="L2_LM_prompt_only"

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for GAMMA in 7e-3
# LEARNING_RATE working: 0.047 0.049 0.052
# GAMMA not working: 1e-4 5e-5 1e-5 5e-6 1e-6 1 5e-1 1e-1 5e-2 1e-2 , some layers work: 5e-3 1e-3 5e-4 1e-5, work: 7e-3(best) 6e-3 5e-3 3e-3 2e-3
do
python3 main_mean_of_prompts_prompt_only.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT \
                             --base_initial=$BASE_INITIAL --similarity=$SIMILARITY
done

# BASELINE_ONLY=False
# PROMPT="trec_1"
# PARTICULAR_LAYER=-2 # 4
# python3 main_mean_of_prompts_prompt_only.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
#                              --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
#                              --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
#                              --seed=$SEED --particular_layer=$PARTICULAR_LAYER --prompt=$PROMPT