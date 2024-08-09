import os
import csv
import json

import numpy as np
import torch
from datasets import load_dataset
import requests

from util import prepro_sentence, prepro_sentence_pair, \
    prepro_sentence_pair_single

def load_data(data_dir, task, k, seed, split):
    data_dir = os.path.join(data_dir, "k-shot", task, "{}-{}".format(k, seed))
    data = []
    if os.path.exists(os.path.join(data_dir, "{}.tsv".format(split))):
        with open(os.path.join(data_dir, "{}.tsv".format(split)), "r") as f:
            for line in f:
                data.append(line.strip().replace("\\n", '\n').split("\t"))
        if task=="CoLA":
            data = [(sent, label) for _, label, _, sent in data]
        elif task=="RTE":
            data = [(json.dumps({
                "text": p, "question": h[:-1] if h.endswith(".") else h
            }), "1" if l=="entailment" else "0")
                    for _, p, h, l in data[1:]]
        elif data[0]==["sentence", "label"]:
            data = data[1:]
    elif os.path.exists(os.path.join(data_dir, "{}.csv".format(split))):
        with open(os.path.join(data_dir, "{}.csv".format(split)), "r") as f:
            for label, text in csv.reader(f):
                data.append((text, label))
    else:
        raise NotImplementedError(data_dir)

    # # all data should have (input, output) format
    assert np.all([len(dp)==2 for dp in data])

    return data


def prepare_data(tokenizer, train_data, test_data, max_length, max_length_per_example,
                 n_classes=2, templates=None, method_type="generative",
                 is_training=False, use_demonstrations=False,
                 ensemble=False, is_null=False):
    
    if type(templates)==list:
        transform = None
        assert len(templates)==n_classes
    else:
        transform = templates
    assert method_type in ["direct", "channel"]

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    '''
    if method==direct, "sent prompt sent prompt ..."
    - prompt should have space
    - if demonstrations are used, 2nd sentneces to the input sentence should have space

    if method==channel, "prompt sent prompt sent ..."
    - input sent should have space
    - if demonstrations are used, 2nd prompts to the input prompt should have space
    '''

    # For calibration method, following Zhao et al. 2021
    if is_null:
        assert test_data is None
        assert method_type=="direct"
        test_data = [("N/A", "0")]

    prefixes_with_space = None
    if transform is None:
        templates = [template.strip() for template in templates]
        if method_type=="direct":
            templates = [" "+template for template in templates]
            if use_demonstrations:
                test_data = [(" "+sent, label) for sent, label in test_data]
        elif method_type=="channel":
            test_data = [(" "+sent, label) for sent, label in test_data]
            if train_data is not None:
                train_data = [(" "+sent, label) for sent, label in train_data]
            prefixes_with_space = [tokenizer(" "+template)["input_ids"] for template in templates]
        else:
            raise NotImplementedError()

    if transform is None:
        test_inputs = [tokenizer(sent)["input_ids"] for sent, _ in test_data]
        truncated = np.sum([len(inputs)>max_length_per_example-16 for inputs in test_inputs])

        if truncated > 0:
            test_inputs = [inputs[:max_length_per_example-16] for inputs in test_inputs]
            print ("%d/%d truncated" % (truncated, len(test_inputs)))

        prefixes = [tokenizer(template)["input_ids"] for template in templates]
        idx = [idx for idx, _prefixes in enumerate(zip(*prefixes))
                if not np.all([_prefixes[0]==_prefix for _prefix in _prefixes])][0]


    else:
        test_inputs = [transform(dp, tokenizer,
                                 max_length_per_example-16,
                                 groundtruth_only=is_training)
                                   for dp in test_data]
        if not is_training:
            assert np.all([len(dp)==2 and
                           np.all([len(dpi)==n_classes for dpi in dp])
                           for dp in test_inputs])


    if is_training:
        assert not use_demonstrations
        assert not ensemble

        input_ids, attention_mask, token_type_ids = [], [], []
        for test_input, dp in zip(test_inputs, test_data):
            if transform is not None:
                test_input, test_output = test_input
                encoded = prepro_sentence_pair_single(
                    test_input, test_output, max_length, bos_token_id, eos_token_id
                )
            else:
                prefix = prefixes[int(dp[1])]
                if method_type=="channel":
                    encoded = prepro_sentence_pair_single(
                        prefix, test_input, max_length, bos_token_id, eos_token_id)
                elif method_type=="direct":
                    encoded = prepro_sentence_pair_single(
                        test_input + prefix[:idx], prefix[idx:], max_length, bos_token_id, eos_token_id)
                else:
                    raise NotImplementedError()
            input_ids.append(encoded[0])
            attention_mask.append(encoded[1])
            token_type_ids.append(encoded[2])
        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids))

    if use_demonstrations:

        if transform is not None:
            raise NotImplementedError()

        if ensemble:
            return prepare_data_for_parallel(
                tokenizer, train_data, test_data,
                max_length, max_length_per_example,
                method_type, n_classes,
                test_inputs, prefixes, idx, prefixes_with_space,
                bos_token_id, eos_token_id)


        assert train_data is not None
        demonstrations = []

        np.random.shuffle(train_data)

        for sent, label in train_data:
            if len(demonstrations)>0:
                if method_type=="direct":
                    sent = " " + sent
                elif method_type=="channel":
                    prefixes = prefixes_with_space

            if transform is None:
                tokens = tokenizer(sent)["input_ids"][:max_length_per_example]
            else:
                tokens = transform(sent, tokenizer, max_length_per_example)
            prefix = prefixes[(int(label))]

            if method_type=="channel":
                tokens = prefix + tokens
            elif method_type=="direct":
                tokens = tokens + prefix
            else:
                raise NotImplementedError()

            demonstrations += tokens

    if transform is None:
        # check if idx is set well
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                assert prefixes[i][:idx]==prefixes[j][:idx]
                assert prefixes[i][idx]!=prefixes[j][idx]

    input_tensors = []

    for i in range(n_classes):
        if transform is None:
            prefix = prefixes[i].copy()
            if method_type=="channel":
                if use_demonstrations:
                    prefix = demonstrations.copy() + prefix
                tensor = prepro_sentence_pair([prefix], test_inputs, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=use_demonstrations)
            elif method_type=="direct":
                if use_demonstrations:
                    prompt = [demonstrations.copy() + test_input + prefix[:idx] for test_input in test_inputs]
                else:
                    prompt = [test_input + prefix[:idx] for test_input in test_inputs]
                tensor = prepro_sentence_pair(prompt,
                                            [prefix[idx:]], max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=use_demonstrations)
            else:
                raise NotImplementedError()
        else:
            input_ids, attention_mask, token_type_ids = [], [], []
            for input_, output_ in test_inputs:
                encoded = prepro_sentence_pair_single(
                    input_[i], output_[i], max_length,
                    bos_token_id,
                    None if is_generation else eos_token_id,
                    allow_truncation=False)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
            tensor = dict(input_ids=torch.LongTensor(input_ids),
                          attention_mask=torch.LongTensor(attention_mask),
                          token_type_ids=torch.LongTensor(token_type_ids))

        input_tensors.append(tensor)


    return input_tensors


def prepare_data_for_parallel(tokenizer, train_data, test_data,
                              max_length, max_length_per_example,
                              method_type, n_classes,
                              test_inputs, prefixes, idx, prefixes_with_space,
                              bos_token_id, eos_token_id):

    # get len(train_data) number of demonstrations

    assert train_data is not None
    demonstrations_list = []

    np.random.shuffle(train_data)

    for sent, label in train_data:
        tokens = tokenizer(sent)["input_ids"][:max_length_per_example]
        prefix = prefixes[(int(label))]
        if method_type=="channel":
            tokens = prefix + tokens
        elif method_type=="direct":
            tokens = tokens + prefix
        else:
            raise NotImplementedError()

        demonstrations_list.append(tokens)

    # check if idx is set well
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            assert prefixes[i][:idx]==prefixes[j][:idx]
            assert prefixes[i][idx]!=prefixes[j][idx]

    input_tensors = []

    for i in range(n_classes):

        if method_type=="channel":
            prefix = prefixes_with_space[i].copy()
            prompt = [demonstrations + prefix
                      for demonstrations in demonstrations_list]
            tensor = prepro_sentence_pair(
                prompt, test_inputs, max_length,
                bos_token_id, eos_token_id,
                allow_truncation=True)

        elif method_type=="direct":
            prefix = prefixes[i].copy()
            prompt = [demonstrations.copy() + test_input + prefix[:idx]
                      for test_input in test_inputs
                      for demonstrations in demonstrations_list]

            tensor = prepro_sentence_pair(prompt,
                                          [prefix[idx:]], max_length,
                                          bos_token_id, eos_token_id,
                                          allow_truncation=True)
        else:
            raise NotImplementedError()

        input_tensors.append(tensor)


    return input_tensors

def load_prompt(prompts_dir, prompt_task, prompt_file_len):
    prompt_files = ["natural_prompts", "good_prompts"]
    if prompt_file_len < 0:
        prompt_files.append("pile")
    else:
        prompt_files.append("pile_n={}".format(prompt_file_len))
    prompts = {}
    for prompt_file in prompt_files:
        with open(os.path.join(prompts_dir, prompt_file+".json"), 'r') as f:
            prompts.update(json.load(f))
    if prompt_task not in prompts:
        raise NotImplementedError()
    return prompts[prompt_task]


def output_metrices(args, dev_results, test_result, prompt, n_prefix, template_idx, return_obj=False):
    metrices = {
        "taskA": args.task,
        "taskB": args.prompt_task,
        "target_prompt": prompt,
        "prompt_f1_threshold": args.f1_threshold,
        "prompt_file_len": args.prompt_file_len,
        "optimize_against_A": args.bad,
        "batch_size": args.batch_size,
        "gamma": args.aux_weight,
        "n_prefix": n_prefix, 
        "template_idx": template_idx,
        "num_training_steps": args.num_training_steps,
        "eval_period": args.eval_period,
        "warmup_steps": args.warmup_steps,
        "seed_results": dev_results, 
        "test_result": test_result,
        "model": args.model_name
    }

    if return_obj:
        return metrices
    else:
        with open(os.path.join(args.out_dir, "{}-{}-metrics.json".format(args.task, args.prompt_task)), 'w') as f:
            json.dump(metrices, f)
    


def load_and_tokenize_data(task, tokenizer,model_name_or_path, data_dir=None):
    
    if task=="SST-2":
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
    elif task=="sst-5":
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
    elif task=="agnews":
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
    elif task=="trec":
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
    elif task=="subj":
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
    elif task=="boolq":
        dataset = load_dataset("google/boolq")
        def create_prompt(example):
            question = example[0]
            passage = example[1]
            prompt = f"Passage: {question} Question: {passage}"
            # print(prompt)
            return prompt

        def preprocess(examples):
            combine_question_and_passage = zip(examples["passage"],examples["question"])
            prompts = [create_prompt(example) for example in combine_question_and_passage]
            outputs = tokenizer(prompts, truncation=True, padding=True, max_length=491 if "bert" in model_name_or_path else None)
            labels = [1 if answer else 0 for answer in examples['answer']]
            outputs['label'] = labels
            return outputs
        
        tokenized_datasets = dataset.map(
            preprocess,
            batched=True,
            remove_columns=['question', 'passage','answer'],
            )
        num_label = 2
    elif task == "cb":
        dataset = load_dataset('super_glue', task)
        def create_prompt(example):
            premise = example[0]
            hypothesis= example[1]
            prompt = f"Premise: {premise} Hypothesis: {hypothesis}"
            # print(prompt)
            return prompt

        def preprocess(examples):
            combine_question_and_passage = zip(examples["premise"],examples["hypothesis"])
            prompts = [create_prompt(example) for example in combine_question_and_passage]
            outputs = tokenizer(prompts, truncation=True, padding=True)
            return outputs
        
        tokenized_datasets = dataset.map(
            preprocess,
            batched=True,
            remove_columns=['premise', 'idx','hypothesis'],
            )
        num_label = 3
    elif task == "copa":
        dataset = load_dataset('super_glue', task)
        def create_prompt(example):
            premise = example[0]
            choice1= example[1]
            choice2= example[2]
            question=example[3]
            prompt = f"Premise: {premise} Choice 1: {choice1} Choice 2: {choice2} Question: {question}"
            # print(prompt)
            return prompt

        def preprocess(examples):
            combine_question_and_passage = zip(examples["premise"],examples["choice1"],examples["choice2"],examples["question"])
            prompts = [create_prompt(example) for example in combine_question_and_passage]
            outputs = tokenizer(prompts, truncation=True, padding=True)
            return outputs
        
        tokenized_datasets = dataset.map(
            preprocess,
            batched=True,
            remove_columns=['premise', 'idx','choice1', 'choice2','question'],
            )
        for i in range(3):
            print(tokenized_datasets['train'][i]['input_ids'])

        num_label = 2
    elif task == "wsc":
        dataset = load_dataset('super_glue', task)
        def create_prompt(example):
            text = example[0]
            span1_text= example[1]
            span2_text= example[2]
            prompt = f"{text} [SEP] {span1_text} [SEP] {span2_text}"
            # print(prompt)
            return prompt

        def preprocess(examples):
            combine_question_and_passage = zip(examples["text"],examples["span1_text"],examples["span2_text"])
            prompts = [create_prompt(example) for example in combine_question_and_passage]
            outputs = tokenizer(prompts, truncation=True, padding=True)
            return outputs
        
        tokenized_datasets = dataset.map(
            preprocess,
            batched=True,
            remove_columns=['text', 'idx','span1_text', 'span2_text','span1_index','span2_index'],
            )
        num_label = 2
        
    print("finishing tokeninzing")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    return tokenized_datasets, num_label

