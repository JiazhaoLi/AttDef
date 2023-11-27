
from genericpath import sameopenfile
from os import posix_fadvise
from pydoc import doc
import random
from urllib.parse import DefragResult
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import codecs
import pickle
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
from PackDataset import packDataset_util_bert, JL_Dataset
from transformers import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import process_model
import argparse
import torch.nn.functional as F

import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import seaborn as sns
import json
from transformers import BertForSequenceClassification as BC
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from transformers import BertTokenizer
# from .. import Transformer_Explainability
from Transformer_Explainability.BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from Transformer_Explainability.BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import ElectraForPreTraining, ElectraTokenizerFast
from sklearn.metrics import accuracy_score
# from captum.attr import (
#     visualization
# )
import torch

def compute_att(model, tokenizer, eval_text_list, trigger_id_set):
    explanations = Generator(model)
    explanations_orig_lrp = Generator(model)
    all_trigger_id_set = sum(trigger_id_set, [])
    method_expl = {"transformer_attribution": explanations.generate_LRP,
                        "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
                        "last_attn": explanations_orig_lrp.generate_attn_last_layer,
                        "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
                        "lrp": explanations_orig_lrp.generate_full_lrp,
                        "rollout": explanations_orig_lrp.generate_rollout}

    print(all_trigger_id_set)

    all_att = []
    all_token = []

    attribute_method = method_expl["partial_lrp"]
    
    highest_token = defaultdict(list)


    for i in tqdm(range(len(eval_text_list))):
        text = eval_text_list[i]
        encoding = tokenizer(text, truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to("cuda")
        attribution_mask = encoding['attribution_mask'].to("cuda")
        expl = attribute_method(input_ids=input_ids, attribution_mask=attribution_mask)[0]
        
        # normalize scores
        expl = expl[1:-1]
        expl = (expl - expl.min()) / (expl.max() - expl.min())
        expl = expl.detach().cpu().numpy()
        input_ids_cpu = input_ids.flatten().detach().cpu().numpy()[1:-1]
        tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu)

        expl_highest_index = np.argmax(expl,axis=0)
        highest_token[tokens[expl_highest_index]].append(i)
        assert len(expl) == len(tokens)
        all_att.append(expl)
        all_token.append(tokens)
    return all_att, all_token

def compute_att_train(model, tokenizer, eval_text_list, trigger_id_set):
    """ For BFClass, we compute the attritbution score for the train set
    """
    explanations = Generator(model)
    explanations_orig_lrp = Generator(model)
    all_trigger_id_set = sum(trigger_id_set, [])
    method_expl = {"transformer_attribution": explanations.generate_LRP,
                        "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
                        "last_attn": explanations_orig_lrp.generate_attn_last_layer,
                        "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
                        "lrp": explanations_orig_lrp.generate_full_lrp,
                        "rollout": explanations_orig_lrp.generate_rollout}
    attribute_method = method_expl["partial_lrp"] # select one of the method
    all_att = []
    all_token = []
    highest_token = defaultdict(list)
    for i in tqdm(range(len(eval_text_list))):
        text = eval_text_list[i]
        encoding = tokenizer(text, truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to("cuda")
        attribution_mask = encoding['attribution_mask'].to("cuda")
        expl = attribute_method(input_ids=input_ids, attribution_mask=attribution_mask)[0]
        # normalize scores
        expl = expl[1:-1]
        expl = (expl - expl.min()) / (expl.max() - expl.min()) # normalize the score
        expl = expl.detach().cpu().numpy()
        input_ids_cpu = input_ids.flatten().detach().cpu().numpy()[1:-1]
        tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu)
        expl_highest_index = np.argmax(expl,axis=0)
        highest_token[tokens[expl_highest_index]].append(i)
        assert len(expl) == len(tokens)
        all_att.append(expl)
        all_token.append(tokens)
    highest_token_sorted = sorted(highest_token.items(), key=lambda x:len(x[1]),reverse=True) # sort the highest fre token
    return all_att, all_token, highest_token_sorted


def masking_triggers(check_state, att_list, text_list, threshold, mask):
    masked_token_list = []
    for i in range(len(text_list)):
        
        tokens = text_list[i]
        expl = att_list[i]
        state = check_state[i]
        tokens_new = tokens.copy()

        expl = (expl - expl.min()) / (expl.max() - expl.min())
        if state == True:
            expl_attack_index_list  = [x[0] for x in enumerate(expl) if x[1] > threshold]
        else:
            expl_attack_index_list  = []
        assert len(expl) == len(tokens)
        
        
        for index in expl_attack_index_list:
            if mask=='None':
                mask = ''
            tokens_new[index] = mask

        """ concatenate the ## subtoken"""
        conact_text = []
        for x in tokens_new:
            if x[:2] =='##' and len(conact_text) > 0:
                conact_text[-1] += x[2:]
            else:
                conact_text.append(x)

        sub_text_list.append(' '.join(conact_text))
        
    return sub_text_list
    
def electra_poison_discriminator(loader, f):
    check_state = []
    from transformers import ElectraForPreTraining, ElectraTokenizerFast
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-large-discriminator").to("cuda")
    dis_tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
    print('start')
    def checking_sen(input_ids):
        discriminator_outputs = discriminator(input_ids)
        token_id_list = input_ids[0].tolist()
        predictions = discriminator_outputs[0].detach().cpu().numpy()
        t = 0
        # print(predictions)
        predictions = [1 if x >=t else 0 for x in predictions[0]]
        # predictions = (torch.sign(discriminator_outputs[0]) + 1) / 2
        # predictions = predictions.tolist()
        suspicious_idx = set([idx for idx in range(len(predictions)) if predictions[idx]==1.0])
        # suspicious_idx = set([idx for idx in range(len(predictions[0])) if predictions[0][idx]==1.0])
        susp_token = [dis_tokenizer.convert_ids_to_tokens(token_id_list[x]) for x in suspicious_idx]
        
        if len(susp_token) > 0:
            return True
        else:
            return False

    # with torch.no_grad():
    check_cnt=0

    for padded_text in tqdm(loader):
        
        input_ids = dis_tokenizer.encode(padded_text, truncation=True, max_length=512, return_tensors="pt").to('cuda')
        check_state.append(checking_sen(input_ids))
        if checking_sen(input_ids):  
            check_cnt += 1
        
        
    print('num apply check',check_cnt,file=f)
    print('num apply check',len(loader))
    print('num apply check',check_cnt/len(loader))
    print('num apply check',check_cnt)
    return check_state 


def Electra_Disc_ours(args, threshold, text_list, poison_label, clean_train_label_list, trigger_word_set):
    """
        We leverage Electra discriminator to check the potential triggers (tokens) by checking the statistical association between the trigger and the label
        Only check the poisoned samples detemined by the sentence-level discriminator label
    """
    
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-large-discriminator").to("cuda")
    dis_tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")

    """convert trigger into ids"""
    all_input_ids_cpu = []
    all_intput_tokens = []
    for token in trigger_word_set:
        input_ids = dis_tokenizer.encode(token, truncation=True, max_length=512, return_tensors="pt").to('cuda')
        input_ids_cpu = input_ids[0].detach().cpu()[1:-1]
        all_input_ids_cpu.append(input_ids_cpu)
        all_intput_tokens.append([dis_tokenizer.convert_ids_to_tokens([input_ids]) for input_ids in input_ids_cpu])
    ###
    """ get the poison index 
    """
    poison_idx_list = []
    for i in range(len(text_list)):
        if poison_label[i] != clean_train_label_list[i]:
            poison_idx_list.append(i)
    print('poison sample _Index', len(poison_idx_list))
    

    all_susp_token_0 = []
    all_susp_token_1 = []
    all_susp_token_all = []
    susp_corspu_idx = []
    all_susp_token_2 = []
    all_susp_token_3 = []
    highest_sample_index = defaultdict(list)
    doc_frequency = defaultdict(int)
    true_positive = 0
    sentence_true_positive = 0
    sentence_positive = 0
    sentence_token_positive = 0
    sentence_flag_all = []
    all_highest_supicisou_tokenid_list = []

    def checking_sen(input_ids):
        """
        Given input sentence 

        """
        discriminator_outputs = discriminator(input_ids)
        token_id_list = input_ids[0].tolist()
        token_id_set = {a for a in token_id_list}
        for token_id in token_id_set:
            doc_frequency[token_id] +=1
        predictions = discriminator_outputs[0].detach().cpu().numpy()
        predictions_score = predictions[0]
        assert len(token_id_list) == len(predictions_score)
        """
            sentence checking: if all tokens <=0, the whole sentence is the suspcisous sentence 
        """
        sentence_flag=False
        all_poison_idx_list=[]
        all_poison_token_id_list = []
        for i in range(len(predictions_score)):
            if predictions_score[i]>=0:
                all_poison_idx_list.append(i)
        if len(all_poison_idx_list) > 0:
            sentence_flag=True
        """ 
            get the highest suspicious score as the trigger word for this sample 
        """
        # highest 
        highest_index = np.argmax(predictions_score)
        highest_susp_token = [token_id_list[highest_index]]
        highest_susp_token_score = [predictions_score[highest_index]]
        # all 
        all_poison_token_id_list = [token_id_list[x] for x in all_poison_idx_list] ## a
        susp_token_sentence = dis_tokenizer.convert_ids_to_tokens([token_id_list[x] for x in all_poison_idx_list])
        return highest_susp_token, highest_susp_token_score, sentence_flag, susp_token_sentence, all_poison_token_id_list
    
    
    

    if not os.path.exists(f'dataset/threshold_{args.corpus}_{trigger_word_set}_{args.poison_rate}_{args.attack_type}.pickle'):
        for idx in tqdm(range(len(text_list))):
            poison_flag = idx in set(poison_idx_list) 
            text = text_list[idx]
            label = poison_label[idx]

            input_ids = dis_tokenizer.encode(text, truncation=True, max_length=512, return_tensors="pt").to('cuda')
            susp_token_id, highest_susp_token_score, sentence_flag, susp_token_sentence, all_poison_token_id_list = checking_sen(input_ids)
            """ all_poison_token_id_list all score >=0 
                    highest_susp_token_score highest

            """
            sentence_flag_all.append(sentence_flag)
            all_highest_supicisou_tokenid_list.append(susp_token_id)
            all_susp_token_all.append(all_poison_token_id_list)

            if sentence_flag:
                susp_corspu_idx.append(idx)
                if label ==0:
                    all_susp_token_0.append(all_poison_token_id_list)
                if label == 1:
                    all_susp_token_1.append(all_poison_token_id_list)
                if label == 2:
                    all_susp_token_2.append(all_poison_token_id_list)
                if label ==3:
                    all_susp_token_3.append(all_poison_token_id_list)
                    
            ## sentence
            if sentence_flag : # if sentence is poison
                sentence_positive+=1
                if idx in set(poison_idx_list) :
                    sentence_true_positive+=1 
        
        print('sentence positive:',sentence_positive )
        print('sentence true positive:',sentence_true_positive )

        with open(f'dataset/threshold_{args.corpus}_{trigger_word_set}_{args.poison_rate}_{args.attack_type}.pickle', 'wb') as handle:
            pickle.dump((sentence_flag_all, doc_frequency, all_highest_supicisou_tokenid_list, all_susp_token_0, all_susp_token_1,all_susp_token_2, all_susp_token_3, all_susp_token_all,susp_corspu_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('poison states existing start read from file')
        with open(f'dataset/threshold_{args.corpus}_{trigger_word_set}_{args.poison_rate}_{args.attack_type}.pickle', 'rb') as handle:
            sentence_flag_all, doc_frequency, all_highest_supicisou_tokenid_list, all_susp_token_0, all_susp_token_1,all_susp_token_2, all_susp_token_3, all_susp_token_all,susp_corspu_idx = pickle.load(handle)
    
    return sentence_flag_all, all_highest_supicisou_tokenid_list, all_susp_token_all,doc_frequency


def poisoned_testing(f, args, trigger_word_set, clean_test, clean_dev, poison_train, poison_dev, poison_test, tokenizer,batch_size, device, criterion, seed, mask):
    
    random.seed(seed)
    print(trigger_word_set)
    trigger_id_set = [tokenizer(trigger_word)['input_ids'][1:-1] for trigger_word in trigger_word_set]
    print('trigger index',trigger_id_set)
    print('trigger word', trigger_word_set)
    
    clean_test_text_list, clean_test_label_list = process_data(clean_test, seed)
    clean_dev_text_list, clean_dev_label_list = process_data(clean_dev, seed)

    poison_train_text_list, poison_train_label_list = process_data(poison_train, seed)
    poison_test_text_list, poison_test_label_list = process_data(poison_test, seed)
    poison_dev_text_list, poison_dev_label_list = process_data(poison_dev, seed)
    

    avg_length = []
    for i in tqdm(range(len(clean_test_text_list))):
        avg_length.append(len(clean_test_text_list[i].split(' ')))
    print('avg', np.mean(avg_length))


    """
       1. First get the ELECTRA sentence level checking 

    """
    clean_check_path = args.clean_check_path
    poison_check_path = args.poison_check_path
    print('clean state store path: ', clean_check_path)
    print('poison state store path: ', poison_check_path)

    
    if not os.path.exists(clean_check_path):
        os.mkdir(clean_check_path)
    if not os.path.exists(poison_check_path):
        os.mkdir(poison_check_path)
    
    if args.attack_type=='clean':
        print('clean states existing start read from file')
        with open(f'{clean_check_path}/states_clean.pickle', 'rb') as handle:
            _, check_state_test_poison = pickle.load(handle)
    else:
        if not os.path.exists(f'{poison_check_path}/states_poison.pickle'):
            print('check corpus')
            check_state_test_poison = checking_corpus(poison_test_text_list,f)
            with open(f'{poison_check_path}/states_poison.pickle', 'wb') as handle:
                pickle.dump(check_state_test_poison, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('poison states existing start read from file')
            with open(f'{poison_check_path}/states_poison.pickle', 'rb') as handle:
                check_state_test_poison = pickle.load(handle)
            
        if not os.path.exists(f'{poison_check_path}/train_states_poison.pickle'):
            print('check corpus')
            check_state_train_poison = checking_corpus(poison_train_text_list,f)
            with open(f'{poison_check_path}/train_states_poison.pickle', 'wb') as handle:
                pickle.dump(check_state_train_poison, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('poison states existing start read from file')
            with open(f'{poison_check_path}/train_states_poison.pickle', 'rb') as handle:
                check_state_train_poison = pickle.load(handle)
    if not os.path.exists(f'{clean_check_path}/states_clean.pickle'):
        print('check corpus')
        check_state_dev_clean = checking_corpus(clean_dev_text_list,f)
        check_state_test_clean = checking_corpus(clean_test_text_list,f)
        with open(f'{clean_check_path}/states_clean.pickle', 'wb') as handle:
            pickle.dump((check_state_dev_clean,check_state_test_clean), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('clean states existing start read from file')
        with open(f'{clean_check_path}/states_clean.pickle', 'rb') as handle:
            check_state_dev_clean,check_state_test_clean = pickle.load(handle)

    if args.attack_type =='clean':
        check_state_test_poison = check_state_test_clean
        poison_test_text_list = clean_test_text_list

    from collections import Counter
    print('dev state',len(clean_dev_text_list),  Counter(check_state_dev_clean), np.round(Counter(check_state_dev_clean)[True] / len(clean_dev_text_list),4) )
    print('test state clean', len(clean_test_text_list), Counter(check_state_test_clean), np.round(Counter(check_state_test_clean)[True] / len(clean_test_text_list),4) )
    print('test poison state', len(poison_test_text_list), Counter(check_state_test_poison), np.round(Counter(check_state_test_poison)[True] / len(poison_test_text_list),4))
    
    """Initial evaluation on model """
    from transformers import BertForSequenceClassification as BC
    model = BC.from_pretrained(model_path, output_attributions=True)
    model = model.to(device)
    
    # we use the clean dev to select the threshold
    _, clean_dev_acc, clean_dev_predict, clean_dev_label ,_ = evaluate(model, tokenizer, clean_dev_text_list, clean_dev_label_list, batch_size, criterion, device)
    initial_dev = accuracy_score(clean_dev_label,clean_dev_predict)
    print(f'Initial Clean Dev ACC:{initial_dev}')
    print(f'Initial Clean Dev ACC:{initial_dev}',file=f)


    """ 
            2. compute the attribution score 
    """
    model_hook = BertForSequenceClassification.from_pretrained(model_path, output_attributions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model_hook = model_hook.to(device)

    clean_test_label_list = [int(x) for x in clean_test_label_list]
    # first compute the clean 
    clean_att_file = model_path + '/attn_clean.pickle'
    if not os.path.exists(clean_att_file):
        print('start clean test and dev compute attribution score')
        all_att_test, all_token_test = compute_att(model_hook, tokenizer, clean_test_text_list, trigger_id_set)
        all_att_dev, all_token_dev = compute_att(model_hook, tokenizer, clean_dev_text_list, trigger_id_set)
        with open(clean_att_file, 'wb') as handle:
            pickle.dump((all_att_test, all_token_test, all_att_dev, all_token_dev), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('read clean attribution score from file ')
        with open(clean_att_file, 'rb') as handle:
            all_att_test, all_token_test, all_att_dev, all_token_dev = pickle.load(handle)
    

    """
        3. loop threshold to defense
    """
    top_k_list = range(100,10,-1)
    for i in tqdm(top_k_list):
        ## clean dev
        c_sub_text_list_dev = construct_sub_corpus_new(check_state_dev_clean, all_att_dev, all_token_dev, i/100, mask)
        _, _, pre_file_dev, label_file_dev,_ = evaluate(model, tokenizer, c_sub_text_list_dev, clean_dev_label_list, batch_size, criterion, device)
        dev_clean_acc_el = accuracy_score(label_file_dev,pre_file_dev)
        delta_dev = np.round(initial_dev - dev_clean_acc_el,4)
        if delta_dev > 0.02: ## 
            break
    if i == 99:
        i=i-1
    return i+1, check_state_train_poison
    
def defense_token(threshold, f, args, check_state_train_poison, trigger_word_set, clean_train, clean_test, clean_dev, poison_dev, poison_test, poison_train, tokenizer,batch_size, device, criterion, seed, mask):
    """
        Given the threshold we get from the clean dev set, we use the threshold to get the potential triggers
    """
    random.seed(seed)    
    trigger_id_set = [tokenizer(trigger_word)['input_ids'][1:-1] for trigger_word in trigger_word_set]
    print('trigger index',trigger_id_set)
    print('trigger word', trigger_word_set)
    
    # 0-positive and 1-negative
    clean_train_text_list, clean_train_label_list = process_data(clean_train, seed)
    clean_test_text_list, clean_test_label_list = process_data(clean_test, seed)
    clean_dev_text_list, clean_dev_label_list = process_data(clean_dev, seed)

    poison_train_text_list, poison_train_label_list = process_data(poison_train, seed)
    poison_test_text_list, poison_test_label_list = process_data(poison_test, seed)
    poison_dev_text_list, poison_dev_label_list = process_data(poison_dev, seed)

    avg_length = []
    for i in tqdm(range(len(clean_test_text_list))):
        avg_length.append(len(clean_test_text_list[i].split(' ')))
    print('avg', np.mean(avg_length))

    print('poison train', len(poison_train_text_list))
    
    model = BC.from_pretrained(model_path, output_attributions=True)
    model = model.to(device)

    """ 
            2. compute the attribution score 
    """
    model_hook = BertForSequenceClassification.from_pretrained(model_path, output_attributions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model_hook = model_hook.to(device)

    clean_test_label_list = [int(x) for x in clean_test_label_list]
    # first compute the clean 
    poison_att_file = model_path + '/poison_atten_train.pickle'
    
    if not os.path.exists(poison_att_file):
        print(poison_att_file)
        print('start compute poison train attribution')
        all_att_train, all_token_train, highest_token_sorted = compute_att_train(model_hook, tokenizer, poison_train_text_list, trigger_id_set)
        with open(poison_att_file, 'wb') as handle:
                pickle.dump((all_att_train, all_token_train,highest_token_sorted), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('poison train attribution read from file')
        with open(poison_att_file, 'rb') as handle:
            all_att_train, all_token_train, highest_token_sorted = pickle.load(handle)

    

    """"""    
    sentence_flag_all , all_highest_supicisou_tokenid_list, all_susp_tokenid,doc_frequency = Electra_Disc_ours(args, threshold, poison_train_text_list, poison_train_label_list, clean_train_label_list, trigger_word_set)

    att_token_doc_frequency_split= defaultdict(int)
    att_token_doc_frequency_0 = defaultdict(int)
    att_token_doc_frequency_1 = defaultdict(int)

    el_token_doc_frequency_split= defaultdict(int)
    el_token_doc_frequency_0 = defaultdict(int)
    el_token_doc_frequency_1 = defaultdict(int)
    

    def get_attribution_suspicious(check_state_train_poison, all_susp_tokenid, all_att_train, all_token_train, poison_train_label_list, threshold):
        sanitized_sentence = []
        ### directly remove all of the potential trigger word 
        print(len(check_state_train_poison),len(all_att_train))
        assert len(check_state_train_poison) == len(all_att_train)
        
        for i in tqdm(range(len(all_att_train))):
            # print(all_att_train[i])
            if check_state_train_poison[i] ==True:
                sen = []
                # print(all_token_train[i])
                for index in range(len(all_att_train[i])):
                    label = poison_train_label_list[i]
                    tokens_above = set([])
                    if all_att_train[i][index] >= threshold/100: 
                        tokens_above.add(all_token_train[i][index])
                    else:
                        sen.append(all_token_train[i][index])
                    for token in list(tokens_above):
                        att_token_doc_frequency_split[token] +=1
                        if label==0:
                            att_token_doc_frequency_0[token] +=1
                        elif label==1:
                            att_token_doc_frequency_1[token] +=1

                for index in range(len(all_susp_tokenid[i])):
                    tokens_above = set([])
                    tokens_above.add(tokenizer.convert_ids_to_tokens(all_susp_tokenid[i][index]))
                    for token in list(tokens_above):
                        el_token_doc_frequency_split[token] +=1
                        if label==0:
                            el_token_doc_frequency_0[token] +=1
                        elif label==1:
                            el_token_doc_frequency_1[token] +=1
            else:
                sen = all_token_train[i]
            sanitized_sentence.append(sen)  

        # print('sanitized', len(sanitized_sentence))
        for k,v in att_token_doc_frequency_split.copy().items():
            att_token_doc_frequency_split[k] = np.max([att_token_doc_frequency_0[k],att_token_doc_frequency_1[k]])
        
        for k,v in el_token_doc_frequency_split.copy().items():
            el_token_doc_frequency_split[k] = np.max([el_token_doc_frequency_0[k],el_token_doc_frequency_1[k]])
        

        return sanitized_sentence, att_token_doc_frequency_split,att_token_doc_frequency_0,att_token_doc_frequency_1,el_token_doc_frequency_split,el_token_doc_frequency_0,el_token_doc_frequency_1
    
    sanitized_sentence, att_token_doc_frequency_split,att_token_doc_frequency_0,att_token_doc_frequency_1,el_token_doc_frequency_split,el_token_doc_frequency_0,el_token_doc_frequency_1 = get_attribution_suspicious(check_state_train_poison, all_susp_tokenid, all_att_train, all_token_train, poison_train_label_list, threshold)

    sanitized_sentence_string = [' '.join(x).replace(' ##','') for x in sanitized_sentence]
    att_token_doc_frequency_split_sorted = sorted(att_token_doc_frequency_split.items(), key=lambda x:x[1],reverse=True)
    
    ## 0.5% of all dataset, we followed the setting BFClass paper, to select the top 0.5% of the dataset
    if args.corpus == 'sst-2':
        lowbound = 30 
    if args.corpus ==  'offenseval':
        lowbound = 80
    if args.corpus == 'ag':
        lowbound = 550
    if args.corpus == 'imdb':
        lowbound = 125
        
    att_token_doc_frequency_split_sorted = [x[0] for x in att_token_doc_frequency_split_sorted if x[1] > lowbound ]

    # print(att_token_doc_frequency_split_sorted)
    with open(args.model_path+'/potential_tokens.tsv','w') as f:
        for i in range(len(att_token_doc_frequency_split_sorted)):
            f.writelines(att_token_doc_frequency_split_sorted[i]+'\t\n')


    return sanitized_sentence, poison_train_label_list

    # for ag we have four classes, for the others, we have binay labels
    highest_0_index= defaultdict(list)
    highest_1_index= defaultdict(list)
    highest_2_index= defaultdict(list)
    highest_3_index= defaultdict(list)
    highest_token_dict = defaultdict(list)
    highest_attribution_token_dict_0= defaultdict(int)
    highest_attribution_token_dict_1= defaultdict(int)


    for i in range(len(all_highest_supicisou_tokenid_list)):
        above_token = []
        for index in range(len(all_att_train)):
            if all_att_train[i][index] > threshold/100: 
                above_token.add(all_token_train[i][index])
    
        if np.max(all_att_dev[i],axis=0) >0: # must the poisoned sample 
            if poison_train_label_list[i] == 1:
                highest_attribution_token_dict_1[highest_token]+= 1
            if poison_train_label_list[i] == 0:
                highest_attribution_token_dict_0[highest_token]+= 1

            """
                highest attension score , and must be the suspicious token  
            """
            if tokenizer(highest_token)['input_ids'][1:-1][0] in set(all_susp_tokenid[i]):
                token_id = all_highest_supicisou_tokenid_list[i][0]
                if poison_train_label_list[i] == 1:
                    highest_1_index[highest_token].append(i)
                if poison_train_label_list[i] == 0:
                    highest_0_index[highest_token].append(i)
                if poison_train_label_list[i] == 2:
                    highest_2_index[highest_token].append(i)
                if poison_train_label_list[i] == 3:
                    highest_3_index[highest_token].append(i)
                highest_token_dict[highest_token].append(i)

    
    LA = defaultdict(int) # this is the implementation of the BFClass paper LA is the annotation in the paper
    LA_diff = defaultdict(int)
    highest_att = defaultdict()
    highest_att_diff = defaultdict()
    for k,v in highest_token_dict.items():
        LA[k] = max(len(highest_0_index[k]),len(highest_1_index[k]),len(highest_2_index[k]),len(highest_3_index[k]))
        LA_diff[k] = max(len(highest_0_index[k]),len(highest_1_index[k]),len(highest_2_index[k]),len(highest_3_index[k])) - min(len(highest_0_index[k]),len(highest_1_index[k]),len(highest_2_index[k]),len(highest_3_index[k]))
        highest_att[k] = max(highest_attribution_token_dict_0[k], highest_attribution_token_dict_1[k])
        highest_att_diff[k] = max(highest_attribution_token_dict_0[k], highest_attribution_token_dict_1[k]) - min(highest_attribution_token_dict_0[k], highest_attribution_token_dict_1[k])


    token_doc_fequency = defaultdict(int)
    for k,v in doc_frequency.items():   
        token_doc_fequency[tokenizer.convert_ids_to_tokens(k)] = v
    # highest attribution is the candidate pools 
    sorted_highest_attribution_token = sorted(highest_att.items(), key=lambda x:x[1],reverse=True)
    sorted_LA = sorted(LA.items(), key=lambda x:x[1],reverse=True)
    

    highest_token_sorted = sorted(LA.items(), key=lambda x:x[1],reverse=True)
    clean_att_file = model_path + '/filter_highest_atten_train_keyword.tsv'
    highest_att = {'keyword':[x[0] for x in highest_token_sorted],'num':[x[1] for x in highest_token_sorted]}
    df = pd.DataFrame(highest_att, columns = ['keyword', 'num'])
    df.to_csv(clean_att_file,sep='\t',index=False)
   
    

        

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='test ASR and clean accuracy')
    parser.add_argument('--model_path', type=str, help='poisoned model path')
    parser.add_argument('--task', type=str, help='task: sentiment or sent-pair')
    parser.add_argument('--corpus', type=str, help='data dir of train and dev file')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--attack_type', type=str, help='trigger word')
    parser.add_argument('--rep_num', type=int, default=3, help='repetitions')
    parser.add_argument('--valid_type', default='acc', type=str, help='metric: acc or f1')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    parser.add_argument('--DATASET', type=str, default='../../dataset', help='dataset')
    parser.add_argument('--subname', type=str, default='base', help='dataset')
    parser.add_argument('--sample', type=str, default='weighted', help='dataset')
    parser.add_argument('--clean_data_path', type=str, default='weighted', help='dataset')
    parser.add_argument('--poison_data_path', type=str, default='weighted', help='dataset')
    parser.add_argument('--clean_check_path', type=str, default='weighted', help='dataset')
    parser.add_argument('--poison_check_path', type=str, default='weighted', help='dataset')
    parser.add_argument('--backbone', type=str, default='weighted', help='dataset')
    parser.add_argument('--record_file', type=str,default='log.log')
    parser.add_argument('--MASK', type=str,default='[MASK]')
    parser.add_argument('--poison_rate',default='0.15')
    parser.add_argument('--check',default='1')
    parser.add_argument('--SEED',default='1')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    
    args = parser.parse_args()
    # print
    attack_type = args.attack_type
    
    ### the dataset path 
    clean_data_path = args.clean_data_path
    poison_data_path = args.poison_data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Trigger word: ", attack_type)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    valid_type = args.valid_type
    criterion = nn.CrossEntropyLoss()
    model_path = args.model_path
    
    model_version = 'bert-base-uncased'
    do_lower_case = True
    f = open(args.record_file, 'w')

    BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    trigger_word_set = [] 

    seed = args.SEED
    clean_train = clean_data_path+'train.tsv'
    clean_test = clean_data_path+'test.tsv'
    clean_dev = clean_data_path+'dev.tsv'
    if args.attack_type!='clean':
        poison_test = poison_data_path+'test.tsv'
        poison_dev = poison_data_path+'dev.tsv'
        poison_train = poison_data_path+'train.tsv'
    else:
        poison_test = clean_data_path+'test.tsv'
        poison_dev = clean_data_path+'dev.tsv'
        poison_train = clean_data_path+'train.tsv'

    print('test file :',poison_train )
    trigger_word_set=[]    
    threshold, check_state_train_poison = poisoned_testing(f, args, trigger_word_set, \
                                                                                    clean_test, clean_dev, poison_train,poison_dev, poison_test, \
                                                                                    tokenizer, BATCH_SIZE, device, \
                                                                                    criterion, args.SEED,args.MASK)
    print('threshold of attribution', threshold) # we use the threshold to filter the attribution score
    sanitized_sentence,poison_train_label_list = defense_token(threshold, f, args, check_state_train_poison, trigger_word_set, clean_train, clean_test, clean_dev, poison_dev, poison_test, poison_train, tokenizer, BATCH_SIZE, device, criterion, args.SEED, args.MASK)

    