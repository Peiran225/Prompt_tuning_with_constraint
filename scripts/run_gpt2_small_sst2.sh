#!/bin/bash
#SBATCH --job-name=sst2  # Specify a name for your job
#SBATCH --output=outputs/sst2_gpt2_small.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/sst2_gpt2_errors.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=32G                  # Memory per node (4GB in this example)
#SBATCH --qos medium

# SBATCH --account cbcb-heng
# SBATCH --partition cbcb-heng


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
GAMMA=1e-8 # 1e-8 upgrade to 0.86 at layer 3 for gpt2 small, 5e-5 for bert
TASK="SST-2"
NUM_OF_INITIAL_TEXT=10000
SEED=42
PROMPT="SST-2_0"
PROMPT_GROUP="TRUE"
cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for SEED in 50 
# GAMMA working: 5e-7 1e-7 5e-8 1e-8 5e-9 1e-9 5e-10 1e-10 1.1e-7(best) 1.2e-7 1.3e-7 1.4e-7
# seed 42: not working very good, but partially work
# seed: 50(better) 100 works
do
    python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
                            --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK --num_of_initial_text=$NUM_OF_INITIAL_TEXT \
                            --seed=$SEED \
                            --prompt=$PROMPT --prompt_groups=$PROMPT_GROUP
                            # --base_initial=$BASE_INITIAL
# --particular_layer=$PARTICULAR_LAYER
done

# for LEARNING_RATE_LM in 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6
# do
#     python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
#                              --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK --num_of_initial_text=$num_of_initial_text
# done

