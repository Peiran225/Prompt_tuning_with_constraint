#!/bin/bash
#SBATCH --job-name=10000-subj  # Specify a name for your job
#SBATCH --output=outputs/gpt2_large_subj.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/gpt2_large_subj.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --mem=64G                  # Memory per node (4GB in this example)
#SBATCH --qos medium


#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng
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
MODEL_PATH="gpt2-large"
GAMMA=5e-8
TASK="subj"
NUM_OF_INITIAL_TEXT=1
PROMPT="subj_0"
PROMPT_GROUP="TRUE"
SEED=90 #

cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for LEARNING_RATE in 1.1e-3 1.2e-3 9e-3 8e-3
# GAMMA not working 1 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7 5e-8 6e-8  4e-8 3e-8 2e-8  
# GAMMA working: 4e-8(not good) 5e-8 (best) 7e-8 8e-8 9e-8 1e-7
# SEED works 65(better) 60 90(best)
do
    python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT --prompt_group=$PROMPT_GROUP
done



