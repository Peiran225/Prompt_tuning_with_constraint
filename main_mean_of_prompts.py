from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoModelForCausalLM
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit, 
    TaskType,
)

from peft.tuners import PromptEmbedding
from datasets import load_dataset
import evaluate
import torch
import numpy as np
import argparse
import logging
import json
import csv
from my_trainer import my_trainer
from transformers import GPT2LMHeadModel,GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss

from MBA import MBA2
from data import load_prompt 
from transformers import set_seed
import wandb

PROMPT_DICT = {
        "NI": ['subtask047_misc_answering_science_questions', 'subtask034_winogrande_question_modification_object',
            'subtask028_drop_answer_generation', 'subtask054_multirc_write_correct_answer',
            'subtask019_mctaco_temporal_reasoning_category', 'subtask021_mctaco_grammatical_logical',
            'subtask027_drop_answer_type_generation', 'subtask038_qasc_combined_fact',
            'subtask029_winogrande_full_object', 'subtask033_winogrande_answer_generation',
            'subtask044_essential_terms_identifying_essential_words', 'subtask050_multirc_answerability',
            'subtask061_ropes_answer_generation', 'subtask002_quoref_answer_generation',
            'subtask037_qasc_generate_related_fact', 'subtask046_miscellaenous_question_typing',
            'subtask057_multirc_classify_incorrect_answer', 'subtask058_multirc_question_answering',
            'subtask006_mctaco_question_generation_transient_stationary',
            'subtask020_mctaco_span_based_question', 'subtask040_qasc_question_generation',
            'subtask042_qasc_incorrect_option_generation',
            'subtask008_mctaco_wrong_answer_generation_transient_stationary',
            'subtask023_cosmosqa_question_generation', 'subtask025_cosmosqa_incorrect_answer_generation',
            'subtask039_qasc_find_overlapping_words', 'subtask045_miscellaneous_sentence_paraphrasing',
            'subtask060_ropes_question_generation', 'subtask007_mctaco_answer_generation_transient_stationary',
            'subtask013_mctaco_answer_generation_absolute_timepoint', 'subtask059_ropes_story_generation',
            'subtask048_multirc_question_generation'],
        "PILE": ['prompt00', 'prompt01', 'prompt02', 'prompt03', 'prompt04', 'prompt05', 'prompt06', 'prompt07',
        'prompt08', 'prompt09', 'prompt10', 'prompt11', 'prompt12', 'prompt13', 'prompt14', 'prompt15',
        'prompt16', 'prompt17', 'prompt18', 'prompt19', 'prompt20', 'prompt21', 'prompt22', 'prompt23',
        'prompt24', 'prompt25', 'prompt26', 'prompt27', 'prompt28', 'prompt29'],
        "TRUE": {
            "SST-2": ["SST-2_0", "SST-2_1", "SST-2_2", "SST-2_3", "SST-2_4"],
            "sst-5": ["sst-5_0", "sst-5_1", "sst-5_2", "sst-5_3", "sst-5_4"],
            "agnews": ["agnews_0", "agnews_1", "agnews_2", "agnews_3", "agnews_4"],
            "trec": ["trec_0", "trec_1", "trec_2", "trec_3", "trec_4"],
            "subj": ["subj_0", "subj_1", "subj_2", "subj_3", "subj_4"]
        },
    }

def main(args):
    
    
    set_seed(args.seed)

    model_name_or_path = args.path 
    
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)


    if any(k in model_name_or_path for k in ("gpt2", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        print("pad token id is none")
        tokenizer.pad_token_id = tokenizer.eos_token_id


    if args.task=="SST-2":
        dataset = load_dataset("sst2")
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["sentence"], padding=True, truncation=True) #, max_length=None)
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence"],
            )
        num_label = 2
    elif args.task=="sst-5":
        dataset = load_dataset("SetFit/sst5")
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["text"], padding=True, truncation=True) #, max_length=None)
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "label_text"],
            )
        num_label = 5
    elif args.task=="agnews":
        dataset = load_dataset("ag_news")
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["text"], padding=True, truncation=True) #, max_length=None)
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            )
        tokenized_datasets['validation'] = tokenized_datasets['test']
        # Delete 'old_key'
        del tokenized_datasets['test']
        num_label = 4
    elif args.task=="trec":
        dataset = load_dataset("trec")
        def rename_column(example):
            example['label'] = example['coarse_label']
            del example['coarse_label']
            return example
        dataset = dataset.map(rename_column)
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["text"], padding=True, truncation=True) #, max_length=None)
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "fine_label"],
            )
        tokenized_datasets['validation'] = tokenized_datasets['test']
        # Delete 'old_key'
        del tokenized_datasets['test']
        num_label = 6
    elif args.task=="subj":
        dataset = load_dataset("SetFit/subj")
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["text"], padding=True, truncation=True) #, max_length=None)
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "label_text"],
            )
        tokenized_datasets['validation'] = tokenized_datasets['test']
        # Delete 'old_key'
        del tokenized_datasets['test']
        num_label = 2

    

    print("finishing tokeninzing")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    
    # Train (original batches 32)
    training_args = TrainingArguments(
        output_dir=model_name_or_path + "-peft-prompt-tuning",
        learning_rate=args.learning_rate, #0.1 has great difference when using LM similarity, but the accuracy with layer -1 is low. 1e-3 if learning rate<=0.01, the projection of soft prompt will always be the original one
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epoch,
        weight_decay=0.01, #0.01
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=args.seed
    )

    training_args_LM = TrainingArguments(
        output_dir=model_name_or_path + "-peft-prompt-tuning",
        learning_rate=args.learning_rate_LM, #0.1 has great difference when using LM similarity, but the accuracy with layer -1 is low. 1e-3 if learning rate<=0.01, the projection of soft prompt will always be the original one
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epoch,
        weight_decay=0.01, #0.01
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=args.seed
    )
    
    # add a hook to track the embeddings of middle layers of model.base_model
    def hook_fn(module, input, output):
        module.embedding_output = output

    # init_text = "What is the sentiment of this sentence? \n Positive , Negative."#"6.00 credit(s) to open a letter from her"
    if args.num_of_initial_text==1:
        prompt_names = [args.prompt]
    else:
        args.prompt_groups = ["TRUE", "NI", "PILE"]
        prompt_names = []
        for prompt_group in args.prompt_groups:
            if prompt_group == "TRUE":
                prompt_names += PROMPT_DICT[prompt_group][args.task]
            else:
                prompt_names += PROMPT_DICT[prompt_group]
    
    init_texts = [(prompt_name, load_prompt(args.prompts_dir, prompt_name, int(args.pile_len))) for prompt_name in
                   prompt_names]
    



    
    if args.base_initial=="Random":
        post_dir = '-gamma-' + str(args.gamma) + '-lr-' + str(args.learning_rate) + '-lr_LM-' + str(args.learning_rate_LM) + '-epoch-' + str(args.epoch) + '-num_of_init_text-' + str(args.num_of_initial_text) + '-seed-' + str(args.seed) + '-random_init_baseline'
    elif args.particular_layer is not None and args.num_of_initial_text is not 1:
        post_dir = '-gamma-' + str(args.gamma) + '-lr-' + str(args.learning_rate) + '-lr_LM-' + str(args.learning_rate_LM) + '-epoch-' + str(args.epoch) + '-num_of_init_text-' + str(args.num_of_initial_text) + '-seed-' + str(args.seed) +'-layer-' + str(args.particular_layer)
    elif args.num_of_initial_text==1: 
        post_dir =  '-' + args.prompt + '-' + '-gamma-' + str(args.gamma) + '-lr-' + str(args.learning_rate) + '-lr_LM-' + str(args.learning_rate_LM) + '-epoch-' + str(args.epoch) + '-num_of_init_text-' + str(args.num_of_initial_text) + '-seed-' + str(args.seed)
    else:
        post_dir = '-gamma-' + str(args.gamma) + '-lr-' + str(args.learning_rate) + '-lr_LM-' + str(args.learning_rate_LM) + '-epoch-' + str(args.epoch) + '-num_of_init_text-' + str(args.num_of_initial_text) + '-seed-' + str(args.seed)

    results_dir = 'results/' + model_name_or_path + '/' + args.task + post_dir + '.csv'
    new_file = True

    peft_config_without_layer = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=10,
        tokenizer_name_or_path=model_name_or_path,
        )
    for init_text_tuple, prompt_name in zip(init_texts, prompt_names): # range(len(init_texts)):
        init_text = init_text_tuple[1]
        

        #for initext in init_texts:
        org_input = tokenizer(init_text
                            , return_tensors='pt')
        num_virtual_tokens = len(org_input['input_ids'][0])
        
        peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text=init_text,
        tokenizer_name_or_path=model_name_or_path,
        )

        
        
        #train with hook on different layer
        if model_name_or_path in ("gpt2", "bert-base-uncased"):
            each_layer = list(range(0,12))
        elif model_name_or_path=="gpt2-medium":
            each_layer = list(range(0,24))
        elif model_name_or_path=="gpt2-large":
            each_layer = list(range(0,36))
        elif model_name_or_path=="llama":
            each_layer = list(range(0,32))
        elif model_name_or_path=="FacebookAI/roberta-base":
            each_layer = list(range(0,12))
        elif model_name_or_path=="facebook/opt-125m":
            each_layer = list(range(0,32))


        a0 = -2
        a = [-2] + [-1] + each_layer
        final_acc_per_prompt = []
        print('baseline_only')
        print(args.baseline_only)
        if args.baseline_only==True:
            aa = [-1]
        elif args.particular_layer >-3:
            aa = [args.particular_layer]
        else:
            aa = a

        for i in aa:
            if i == -2:
                print("train the model when the hook layer is %s"% i)
                g_p = init_text #"What is the sentiment of this sentence? \n Positive , Negative."
                tokenized_g_p = tokenizer(g_p, padding="max_length", truncation=True, max_length=num_virtual_tokens)
                tokenized_g_p['input_ids'] = torch.tensor(tokenized_g_p['input_ids'])

                # print("training with penalized model")
                model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, cache_dir='/fs/nexus-scratch/peiran/.cache', num_labels=num_label)
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
                    
                if any(k in model_name_or_path for k in ("gpt", "bert", "llama")):
                    model.config.pad_token_id = tokenizer.pad_token_id
        
                trainer = my_trainer(
                    model=model,
                    args=training_args_LM,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    model_name_or_path=model_name_or_path,
                    tokenized_g_p=tokenized_g_p,
                    hook_layer=i,
                    similarity="L2",
                    gamma=args.gamma
                ) 

                trainer.train()
                print("trainer.metrics_log %s"% trainer.metrics_log)
                final_acc_per_prompt.append(trainer.metrics_log[-1]['eval_accuracy'])
                outputs_list = trainer.metrics_log
                outputs_list = [{**d, 'layer': i, 'prompt': prompt_name} for d in outputs_list]
                if new_file == True:
                    headers = outputs_list[0].keys()
                    with open(results_dir, 'w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=headers)
                        # Write the headers (column names)
                        writer.writeheader()
                        for d in outputs_list:
                            writer.writerow(d)
                    new_file = False
                    print("new file %s"% new_file)
                 
                else:
                    print(new_file)
                    with open(results_dir, 'a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=headers)
                        for d in outputs_list:
                            writer.writerow(d)
            elif i == -1:
                print("train the model when the hook layer is %s"% i)
                # wandb.init(project='prompting' + args.task, 
                #            entity='pyu123',
                #             config={ "model": args.path,
                #                         "learning_rate": args.learning_rate,
                #                         "learning_rate_LM": args.learning_rate_LM,
                #                         "gamma": args.gamma,
                #                         "epochs": args.epoch,
                #                         "layer": i
                #                         })
                model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=num_label)
                if args.base_initial=="Random":
                    model = get_peft_model(model, peft_config_without_layer)
                else:
                    model = get_peft_model(model, peft_config)
                
                

                if any(k in model_name_or_path for k in ("gpt",)):
                    # print(model_name_or_path)
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "bert-base-uncased":
                    model.base_model.bert.encoder.layer[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "EleutherAI/gpt-j-6b":
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "FacebookAI/roberta-base":
                    model.base_model.base_model.encoder.layer[i].register_forward_hook(hook_fn) 
                    
                if any(k in model_name_or_path for k in ("gpt", "bert", "llama")):
                    model.config.pad_token_id = tokenizer.pad_token_id
                

                trainer = my_trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics
                    # gamma=1e-4 #1e-4 will let aux_loss be in 1/10 of loss (around 0.8) at the beginning.
                ) 
                # eval = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
                # "Accuracy of projected soft prompt before train\n %s"% eval)


                trainer.train()
                print("trainer.metrics_log %s"% trainer.metrics_log)
                outputs_list = trainer.metrics_log
                outputs_list = [{**d, 'layer': i, 'prompt': prompt_name} for d in outputs_list]
                with open(results_dir, 'a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    for d in outputs_list:
                        writer.writerow(d)
            else:
                print("train the model when the hook layer is %s"% i)
                # wandb.init(project='prompting' + args.task, 
                #            entity='pyu123',
                #             config={ "model": args.path,
                #                         "learning_rate": args.learning_rate,
                #                         "learning_rate_LM": args.learning_rate_LM,
                #                         "gamma": args.gamma,
                #                         "epochs": args.epoch,
                #                         "layer": i
                #                         })
                # compute the hook of the good_prompt
                g_p = init_text #"What is the sentiment of this sentence? \n Positive , Negative."
                tokenized_g_p = tokenizer(g_p, padding="max_length", truncation=True, max_length=num_virtual_tokens)
                tokenized_g_p['input_ids'] = torch.tensor(tokenized_g_p['input_ids'])

                # print("training with penalized model")
                model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, cache_dir='/fs/nexus-scratch/peiran/.cache', num_labels=num_label)
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

                # add the hook to the ith layer of the model
                if any(k in model_name_or_path for k in ("gpt",)):
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "bert-base-uncased":
                    model.base_model.bert.encoder.layer[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "EleutherAI/gpt-j-6b":
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "FacebookAI/roberta-base":
                    model.base_model.base_model.encoder.layer[i].register_forward_hook(hook_fn) 
                    
                if any(k in model_name_or_path for k in ("gpt", "bert", "llama")):
                    model.config.pad_token_id = tokenizer.pad_token_id
        
                trainer = my_trainer(
                    model=model,
                    args=training_args_LM,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    model_name_or_path=model_name_or_path,
                    tokenized_g_p=tokenized_g_p,
                    hook_layer=i,
                    similarity="L2_LM",
                    gamma=args.gamma
                ) 
                    
                trainer.train()
                print("trainer.metrics_log %s"% trainer.metrics_log)
                outputs_list = trainer.metrics_log
                outputs_list = [{**d, 'layer': i, 'prompt': prompt_name} for d in outputs_list]
                with open(results_dir, 'a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    for d in outputs_list:
                        writer.writerow(d)
                    
        if args.num_of_initial_text == 1:
            break
                
        



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=str, default = None)
    parser.add_argument("--log_file", default=None, type=str)
    parser.add_argument("--path", default="FacebookAI/roberta-base", type=str)
    parser.add_argument("--hook_layer", default=-1, type=int)
    parser.add_argument("--prompts_dir", default="/fs/nexus-scratch/peiran/prompting_with_constraints/prompts", type=str)
    parser.add_argument("--prompt_groups", default=["TRUE", ], type=list)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--task", default="trec", type=str)
    # parser.add_argument("--dataset", default="trec", type=str)
    parser.add_argument("--pile_len", default=-1, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--learning_rate_LM", default=0.01, type=float)
    parser.add_argument("--gamma", default=1e-5, type=float)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_of_initial_text", default=None, type=int)
    parser.add_argument("--particular_layer", default=-3, type=int)
    parser.add_argument("--baseline_only", default=False, type=bool)
    parser.add_argument("--base_initial", default="Text", type=str)

    args = parser.parse_args()
    
    # args.learning_rate = 1e-3
    # args.learning_rate_LM = 1e-3
    # args.gamma = 1e-8 # 1e-8 upgrade to 0.86 at layer 3 for gpt2 small, 5e-5 for bert
    # args.epoch = 1
    # args.path = "gpt2" #bert-base-uncased
    print(args)

     #gpt2=gpt small
    main(args)





