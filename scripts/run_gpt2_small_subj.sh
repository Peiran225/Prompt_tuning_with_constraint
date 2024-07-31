#!/bin/bash
#SBATCH --job-name=subj  # Specify a name for your job
#SBATCH --output=outputs/subj_gpt2_small.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/subj_gpt2_small.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=32G                  # Memory per node (4GB in this example)
#SBATCH --qos medium

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

CUDA_VISIBLE_DEVICES=0
# fedadmm
# for LEARNING_RATE in 0.012 0.014 0.016 0.02
# for DATASET in 'synthetic_1_1' 'synthetic_0_0' 'synthetic_0.5_0.5' 'FEMNIST'
# do
#     EXP_ID=$EXP_ID_BASE$LEARNING_RATE
#     python main.py --dataset=$DATASET --optimizer=$OPTIMIZER --exp_id=$EXP_ID  \
#                 --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
#                 --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 \
#                 --batch_size=$BATCH_SIZE --num_epochs=$NUM_EPOCH --model=$MODEL --local_optim=$LOCAL_OPTIM \
#                 --term_alpha=$TERM_ALPHA
# done

LEARNING_RATE=1e-3 # 1e-3 work
LEARNING_RATE_LM=1e-3
EPOCH=1
MODEL_PATH="gpt2"
GAMMA=4e-8 # 
TASK="subj"
NUM_OF_INITIAL_TEXT=10000
PROMPT="subj_0"
PROMPT_GROUP="TRUE"
SEED=42 

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for SEED in 50 55 60 65 70 75 80 85 90 95 100 200
# GAMMA not working 1 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7 5e-8 
# GAMMA working: 4e-8(best) 5e-8 1e-7 4e-8 3e-8 2e-8 6e-8 7e-8 8e-8 9e-8
# SEED 
do
    python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT 
done



