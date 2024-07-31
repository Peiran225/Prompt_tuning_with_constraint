import numpy as np

loaded_data = np.load('/fs/nexus-scratch/peiran/prompting_with_constraints/peft_experiments/scripts/outputs/outputs_gpt2_small/acc_of_different_prompts_with_layers.npy')

print(loaded_data)
