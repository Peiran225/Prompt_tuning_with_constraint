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
from my_trainer import my_trainer
from transformers import GPT2LMHeadModel,GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss

from MBA import MBA2
from data import load_prompt
import logging


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
    
    
    

    print("finishing tokeninzing")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    t = tokenized_datasets['train'][0]
    
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # init_text = "What is the sentiment of this sentence? \n Positive , Negative."#"6.00 credit(s) to open a letter from her"
    if args.prompt_group == "TRUE":
        prompt_names = PROMPT_DICT[args.prompt_group][args.task]
    init_texts = [(prompt_name, load_prompt(args.prompts_dir, prompt_name, int(args.pile_len))) for prompt_name in
                   prompt_names]
    init_text = init_texts[0][1]
    # org_input = tokenizer('What is the sentiment of this sentence? \n Positive , Negative.'
    #                       , return_tensors='pt')
    org_input = tokenizer(init_text
                           , return_tensors='pt')
    # org_input = tokenizer('6.00 credit(s) to open a letter from her', return_tensors='pt')
    num_virtual_tokens = len(org_input['input_ids'][0])
    # peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=num_virtual_tokens,
    prompt_tuning_init_text=init_text,
    tokenizer_name_or_path=model_name_or_path,
)
    


    
    


    def map_to_discrete(p_embeddings, model):
        indices = []
        aux_loss = 0
        volcabulary_weight = model.word_embeddings.weight
        for vector in p_embeddings:
            distances = torch.linalg.norm(vector - volcabulary_weight.cpu(), dim=1)
            d, i = torch.sort(distances)
            indices.append(i[0].item())
            aux_loss += d[0] ** 2
        weight = volcabulary_weight[indices]
        # model.new_embed._load_from_state_dict({"weight": weight},
                                            # "", None, True, [], [], "")
        return indices, aux_loss
    
    
    

    # Train (original batches 32)
    training_args = TrainingArguments(
        output_dir="gpt2-peft-prompt-tuning",
        learning_rate=args.learning_rate, #0.1 has great difference when using LM similarity. 1e-3 if learning rate<=0.01, the projection of soft prompt will always be the original one
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01, #0.01
        load_best_model_at_end=True,
        evaluation_strategy="steps",
		save_strategy="steps",
		eval_steps=100,
        seed=42,
        data_seed=42
    )
    
    # add a hook to track the embeddings of middle layers of model.base_model
    def hook_fn(module, input, output):
        module.embedding_output = output
    
    #train with hook on different layer
    each_layer = list(range(0,12))
    a0 = -2
    a = [-2] + [-1] + each_layer
    for i in [-1]:
        if i == -2:
            print("train the model when the hook layer is %s"% i)
            peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text=init_text,
            tokenizer_name_or_path=model_name_or_path,
            )
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            if model_name_or_path == "gpt2":
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
                # similarity="L2_LM",
            # gamma=1e-4 
            

            trainer.train()
        elif i == -1:
            print("train the model when the hook layer is %s"% i)
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=5)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            p_embeddings = model.get_prompt_embedding_to_save('default')
            # print(p_embeddings)
            
            
            # # interpret with projection
            # interpretation_proj, _ = map_to_discrete(p_embeddings, model)
            # # print(interpretation_proj)
            # # print(tokenizer.convert_ids_to_tokens(interpretation_proj))
            # # interpret with LM 
            # # torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            # causal_model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)#.to(torch_device)
            
            # # generate with the original text
            # output_org = causal_model.generate(**org_input, max_new_tokens=num_virtual_tokens)
            # # print(output_org)
            # generated_text_org = tokenizer.decode(output_org[0], skip_special_tokens=True)
            # # print("input_ids of org_input: %s"% org_input["input_ids"][0])
            # # print("The output_org  with LM generate function is: \n{}".format(generated_text_org))
            
            # #generate with the soft prompts
            # causal_model.resize_token_embeddings(len(tokenizer) + num_virtual_tokens)
            # embeddings_causal_model = causal_model.transformer.wte.weight
            # embeddings_causal_model.data[-num_virtual_tokens:, :] = p_embeddings
            # p = {'input_ids': torch.tensor([[i for i in range(50257, 50257 + num_virtual_tokens)]]), 
            #     'attention_mask': torch.tensor([[1 for i in range(50257, 50257 + num_virtual_tokens)]])}
            # # print("input_ids of p: %s"% p["input_ids"][0])
            # output = causal_model.generate(**p, max_new_tokens=num_virtual_tokens)
            # # print(output)
            # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # # print("The output  with LM generate function is: \n{}".format(generated_text))


            if model_name_or_path == "gpt2":
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
                # similarity="L2_LM",
            # gamma=1e-4 
            # eval = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
            # print("Accuracy of projected soft prompt before train\n %s"% eval)


            trainer.train()
            print("trainer.metrics_log %s"% trainer.metrics_log)
        else:
            print("train the model when the hook layer is %s"% i)
            # compute the hook of the good_prompt
            g_p = init_text #"What is the sentiment of this sentence? \n Positive , Negative."
            tokenized_g_p = tokenizer(g_p, padding="max_length", truncation=True, max_length=num_virtual_tokens)
            tokenized_g_p['input_ids'] = torch.tensor(tokenized_g_p['input_ids'])
        
            # print("training with MBA")
            # model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
            # model = get_peft_model(model, peft_config)
            # model.print_trainable_parameters()

            # p_embeddings = model.get_prompt_embedding_to_save('default')


            # if model_name_or_path == "gpt2":
            #     model.config.pad_token_id = tokenizer.pad_token_id 
            


            # # add the hook to the first layer of the model
            # if model_name_or_path == "gpt2": 
            #     model.base_model.transformer.h[i].register_forward_hook(hook_fn)
            



            
        
            # if args.path in ("gpt2", "bert-base-uncased"):
            #     model.config.pad_token_id = tokenizer.pad_token_id
    
            # # epoch 32* data point 6000* batchsize
            # trainer = my_trainer(
            #     model=model,
            #     args=training_args,
            #     train_dataset=tokenized_datasets["train"],
            #     eval_dataset=tokenized_datasets["validation"],
            #     tokenizer=tokenizer,
            #     data_collator=data_collator,
            #     compute_metrics=compute_metrics,
            #     model_name_or_path=model_name_or_path,
            #     tokenized_g_p=tokenized_g_p,
            #     hook_layer=i,
            #     my_optimizer_name="MBA2",
            #     my_optim=MBA2,
            #     similarity="L2_LM",
            #     delta = 1e10,
            #     # the distance in the beginning is 6e7, delta>=1e2 is the same acc with unconstraint
            # ) 
            #     # similarity="L2_LM",
            # # gamma=1e-4 
            # # eval = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
            # # print("Accuracy of projected soft prompt before train\n %s"% eval)


            # trainer.train()

            print("training with penalized model")
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, cache_dir='/fs/nexus-scratch/peiran/.cache', num_labels=5)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

            # add the hook to the first layer of the model
            if any(k in model_name_or_path for k in ("gpt")):
                model.base_model.transformer.h[i].register_forward_hook(hook_fn)
            elif model_name_or_path == "bert-base-uncased":
                model.base_model.bert.encoder.layer[i].register_forward_hook(hook_fn)
            elif model_name_or_path == "EleutherAI/gpt-j-6b":
                model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                
            if any(k in model_name_or_path for k in ("gpt", "bert", "llama")):
                model.config.pad_token_id = tokenizer.pad_token_id
    
            trainer1 = my_trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                model_name_or_path=model_name_or_path,
                tokenized_g_p=tokenized_g_p,
                hook_layer=i,
                similarity="L2_LM",
                gamma=5e-5
                # gamma=1e-4 #1e-4 will let aux_loss be in 1/10 of loss (around 0.8) at the beginning.
            ) 
                # similarity="L2_LM",
            # gamma=1e-4 
            trainer1.train()


        # # extract the output of the embeddings of hte soft prompt:
        # p_embeddings_after_train = model.get_prompt_embedding_to_save('default')
        # # print(p_embeddings_after_train)
        # interpretation_proj_after_train, _ = map_to_discrete(p_embeddings_after_train.cpu(), model)
        # # print(interpretation_proj_after_train)
        # projected_tokens = tokenizer.convert_ids_to_tokens(interpretation_proj_after_train)
        # print("the projected tokens after train are \n%s"% projected_tokens)
        # #assign the projected prompt to the model
        
        # # model._peft_config = {"default": peft_config_after}
        # model_eval = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
        # model_eval = get_peft_model(model_eval, peft_config_after)
        # # model.prompt_encoder = PromptEmbedding(peft_config_after, model.word_embeddings)
        # if args.path in ("gpt2", "bert-base-uncased"):
        #         model_eval.config.pad_token_id = tokenizer.pad_token_id

        # eval_model = my_trainer(
        #     model=model_eval,
        #     args=training_args,
        #     train_dataset=tokenized_datasets["train"],
        #     eval_dataset=tokenized_datasets["validation"],
        #     tokenizer=tokenizer,
        #     data_collator=data_collator,
        #     compute_metrics=compute_metrics
        # ) 
        # eval_after = eval_model.evaluate(eval_dataset=tokenized_datasets["validation"])
        # print("Accuracy of projected soft prompt after train:\n %s"% eval_after)













if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=str, default = None)
    parser.add_argument("--log_file", default=None, type=str)
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--hook_layer", default=-1, type=int)
    parser.add_argument("--prompts_dir", default="/fs/nexus-scratch/peiran/prompting_with_constraints/prompts", type=str)
    parser.add_argument("--prompt_group", default="TRUE", type=str)
    parser.add_argument("--task", default="sst-5", type=str)
    parser.add_argument("--pile_len", default=-1, type=int)
  
    args = parser.parse_args()
    

    # handlers = [logging.StreamHandler()]
    # if args.log_file is not None:
    #     handlers.append(logging.FileHandler(args.log_file))
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO,
    #                     handlers=handlers)
    # logger = logging.getLogger(__name__)
    # logger.info(args)
    # prefix_dir = '.'
    # prefix = 'sst5' + ''
    # postfix = args.prompt_group + str(args.lr) 
    # logger = Logger(prefix_dir + '/gpt2_small', prefix = prefix, postfix= postfix)


    args.path = "gpt2" #gpt2=gpt small
    for lr in (2, 1e-2, 1e-2, 1e-6, 1e-7, 1e-8):
        args.learning_rate = lr
        main(args)





