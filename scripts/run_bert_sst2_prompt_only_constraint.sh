#!/bin/bash
#SBATCH --job-name=B-sst2 # Specify a name for your job
#SBATCH --output=outputs/bert_sst2.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/bert_sst2.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
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

LEARNING_RATE=1e-3 # 0.1， 0.01 work
LEARNING_RATE_LM=1e-3
EPOCH=1
MODEL_PATH="bert-base-uncased" # "gpt2"
GAMMA=5e-7 # 1e-8 upgrade to 0.86 at layer 3 for gpt2 small, 5e-5 for bert
TASK="SST-2"
PROMPT_GROUP="TRUE"
NUM_OF_INITIAL_TEXT=10000
SEED=50 #42 not working.
# BASELINE_ONLY=True
PROMPT="SST-2_3"
SIMILARITY="L2_LM_prompt_only"



cd /fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments

for SEED in 50 
# GAMMA not working 1e-3 5e-4 1e-4 5e-5 1e-5 1e-6 1e-7 5e-8 1e-8 working: 5e-7
# SEED 47 50(best) works 
do
python3 main_mean_of_prompts_prompt_only.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE_LM  \
                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK \
                             --num_of_initial_text=$NUM_OF_INITIAL_TEXT --baseline_only=$BASELINE_ONLY \
                             --seed=$SEED --prompt=$PROMPT \
                             --base_initial=$BASE_INITIAL --similarity=$SIMILARITY
done


# NUM_OF_INITIAL_TEXT=1
# PROMPT="SST-2_3"
# PARTICULAR_LAYER=-2
# python3 main_mean_of_prompts.py --learning_rate=$LEARNING_RATE --learning_rate_LM=$LEARNING_RATE  \
#                             --epoch=$EPOCH --path=$MODEL_PATH --gamma=$GAMMA --task=$TASK --num_of_initial_text=$NUM_OF_INITIAL_TEXT \
#                             --seed=$SEED --baseline_only=$BASELINE_ONLY \
#                             --particular_layer=$PARTICULAR_LAYER \
#                             --prompt_group=$PROMPT_GROUP --prompt=$PROMPT \
#                             # --base_initial=$BASE_INITIAL