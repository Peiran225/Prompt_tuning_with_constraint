#!/bin/bash
#SBATCH --job-name=trec  # Specify a name for your job
#SBATCH --output=outputs/trec_gpt2_small.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/trec_gpt2_small.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
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

LEARNING_RATE=0.05 # 1e-3 work
LEARNING_RATE_LM=0.05
EPOCH=2 #1 does not work
MODEL_PATH="gpt2"
GAMMA=1e-5 # 
TASK="trec"
NUM_OF_INITIAL_TEXT=10000
PROMPT="trec_0"
PROMPT_GROUP="TRUE"
SEED=42
cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

# for LEARNING_RATE in 0.05
# # LEARNING RATE 0.01 0.03 0.04 0.05 0.08 work, 1 1e-1  1e-3 1e-4 5e-5 1e-6 5e-7 1e-7 5e-8 not work.
# do
#     python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
#                              --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK --num_of_initial_text=$NUM_OF_INITIAL_TEXT
# done


for GAMMA in 1e-6
# GAMMA working: 5e-5 1e-6(best) 3e-6 4e-6 5e-6 not working: 1e-3
do
python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT \
                             --seed=$SEED  --prompt=$PROMPT \
                             
done

