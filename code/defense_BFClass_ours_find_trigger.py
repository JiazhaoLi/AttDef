
from os import posix_fadvise
import random
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import codecs
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import process_model
import argparse
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from transformers import ElectraForPreTraining, ElectraTokenizerFast

from transformers import BertTokenizer
# from .. import Transformer_Explainability
from Transformer_Explainability.BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from Transformer_Explainability.BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from transformers import AutoTokenizer
from collections import Counter
import itertools



def compute_att(model, tokenizer, eval_text_list, trigger_id_set, poison_test_index_list):
    explanations = Generator(model)
    explanations_orig_lrp = Generator(model)
    method_expl = {"transformer_attribution": explanations.generate_LRP,
                        "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
                        "last_attn": explanations_orig_lrp.generate_attn_last_layer,
                        "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
                        "lrp": explanations_orig_lrp.generate_full_lrp,
                        "rollout": explanations_orig_lrp.generate_rollout}

    all_att = []
    all_token = []
    poison_index = []
    attribute_method = method_expl["partial_lrp"]
    for i in tqdm(range(len(eval_text_list))):
        text = eval_text_list[i]
        trigger_inex = poison_test_index_list[i]
        word_list = text.split(' ')
        new_index = []
        for trigger in trigger_inex:
            word_list_pre = word_list[:trigger]
            encoding_pre = tokenizer(' '.join(word_list_pre), truncation=True, return_tensors='pt')['input_ids']
            new_index.append(len(encoding_pre[0][1:-1]))

        encoding = tokenizer(text, truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to("cuda")
        attention_mask = encoding['attention_mask'].to("cuda")
        expl = attribute_method(input_ids=input_ids, attention_mask=attention_mask)[0]

        expl = expl[1:-1]
        # expl = (expl - expl.min()) / (expl.max() - expl.min()) # we leave the normalization to the threshold selection step
        expl = expl.detach().cpu().numpy()
        input_ids_cpu = input_ids.flatten().detach().cpu().numpy()[1:-1]
        tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu)
        assert len(expl) == len(tokens)
        all_att.append(expl)
        all_token.append(tokens)
        poison_index.append(new_index)
    return all_att, all_token, poison_index



def masking_triggers(prefilter_trigger_word_set, check_state, att_list, text_list, trigger_list_all, threshold, mask):

    masked_token_list = []
    prefilter_trigger_id_set = [tokenizer.convert_tokens_to_ids(x) for x in prefilter_trigger_word_set]

    pre_filter_remove_GT = [] # prefiltered ground truth trigger word
    pre_filter_remove_FP = [] # prefiltered false positive trigger word
    att_filter_remove_GT = [] # computed ground truth trigger word 
    att_filter_remove_FP = [] # computed false positive trigger word

    assert len(trigger_list_all) == len(text_list)
    for i in range(len(text_list)):
        remove_tokens = []
        tokens = text_list[i]  # this is subtoken level 
        expl = att_list[i]
        trigger_idx = trigger_list_all[i]
        state = check_state[i]
        tokens_new = tokens.copy()
        toknen_ids_list = [tokenizer.convert_tokens_to_ids(x) for x in tokens_new]
        pre_filter_index_list_found = [x[0] for x in enumerate(toknen_ids_list) if x[1] in prefilter_trigger_id_set] # remove based on prefiltering 
        pre_filter_id_list_found = [x[1] for x in enumerate(toknen_ids_list) if x[1] in prefilter_trigger_id_set] # remove based on prefiltering 
        pre_filter_found_GT = set(pre_filter_index_list_found).intersection(set(trigger_idx))
        pre_filter_remove_GT.append(pre_filter_found_GT)

        pre_filter_found_FP = [x for x in pre_filter_index_list_found if x not in set(trigger_idx) ]
        pre_filter_remove_FP.append(set(pre_filter_found_FP))

        ########## attribution based set trigger removal  
        expl = (expl - expl.min()) / (expl.max() - expl.min())
        if state == True: # this is identified as poisoned sample 
            expl_attack_index_list  = [x[0] for x in enumerate(expl) if x[1] > threshold]
        else:
            expl_attack_index_list  = []  # get the indexed list to be removed 
        
        att_removal_GT = set(expl_attack_index_list).intersection(set(trigger_idx))
        att_filter_remove_GT.append(att_removal_GT)

        att_removal_FP = [x for x in expl_attack_index_list if x not in set(trigger_idx)]
        att_filter_remove_FP.append(set(att_removal_FP))
        assert len(expl) == len(tokens)
        for index in expl_attack_index_list : 
            tokens_new[index] = mask
            remove_tokens.append(tokens_new[index])
        

        """ concatenate the ## subtoken"""
        conact_text = []
        for x in tokens_new:
            if x[:2] =='##' and len(conact_text) > 0:
                conact_text[-1] += x[2:]
            else:
                conact_text.append(x)
        masked_token_list.append(' '.join(conact_text))

   

    assert len(pre_filter_remove_GT) == len(pre_filter_remove_FP)
    assert len(att_filter_remove_GT) == len(att_filter_remove_FP)

    return masked_token_list, pre_filter_remove_GT, pre_filter_remove_FP, att_filter_remove_GT,att_filter_remove_FP
    

def electra_poison_discriminator(loader, f):

    check_state = []
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-large-discriminator",cache_dir=cache).to("cuda")
    dis_tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator",cache_dir=cache)
    print('start')
    def checking_sen(input_ids):
        discriminator_outputs = discriminator(input_ids)
        token_id_list = input_ids[0].tolist()
        predictions = discriminator_outputs[0].detach().cpu().numpy()
        t = 0

        predictions = [1 if x >=t else 0 for x in predictions[0]]
        suspicious_idx = set([idx for idx in range(len(predictions)) if predictions[idx]==1.0])
        susp_token = [dis_tokenizer.convert_ids_to_tokens(token_id_list[x]) for x in suspicious_idx]
        if len(susp_token) > 0:
            return True
        else:
            return False

    check_cnt=0
    for padded_text in tqdm(loader):
        input_ids = dis_tokenizer.encode(padded_text, truncation=True, max_length=512, return_tensors="pt").to('cuda')
        check_state.append(checking_sen(input_ids))
        if checking_sen(input_ids):  
            check_cnt += 1
    return check_state 

def evaluate_trigger(args, pre_filter_remove_GT, pre_filter_remove_FP, att_filter_remove_GT, att_filter_remove_FP):
    pre_cnt_GT = 0
    pre_cnt_FP = 0
    att_cnt_GT = 0
    att_cnt_FP = 0

    full_remove_pre = 0
    full_remove_att = 0
    scalce = 3
    if args.attack_type=='sentence':
        scalce=5
    for i in range(len(pre_filter_remove_GT)):
        pre_cnt_GT += len(pre_filter_remove_GT[i])
        att_cnt_GT += len(att_filter_remove_GT[i])
        pre_cnt_FP += len(pre_filter_remove_FP[i])
        att_cnt_FP += len(att_filter_remove_FP[i])
        if len(pre_filter_remove_GT[i]) ==scalce:
            full_remove_pre+=1
        if len(att_filter_remove_GT[i]) ==scalce:
            full_remove_att+=1

    pre_remove_GT = pre_cnt_GT / len(pre_filter_remove_GT) /scalce
    att_remove_GT = att_cnt_GT / len(att_filter_remove_GT) /scalce
    FP_pre_avg_remove = pre_cnt_FP / len(pre_filter_remove_GT) 
    FP_att_avg_remove = att_cnt_FP/ len(att_filter_remove_GT)
    return pre_remove_GT, att_remove_GT, FP_pre_avg_remove, FP_att_avg_remove, full_remove_pre / len(pre_filter_remove_GT) , full_remove_att / len(att_filter_remove_GT)

   
def poisoned_testing(f, args, all_trigger_word_set, test_file, dev_file, poison_dev, poison_file, tokenizer,batch_size, device, criterion, seed, mask):
    
    random.seed(seed)
    ground_truth_trigger, trigger_word_set = all_trigger_word_set
    print(trigger_word_set)
    

    train_pre_trigger_id_set = [tokenizer(trigger_word)['input_ids'][1:-1] for trigger_word in trigger_word_set]
    train_pre_trigger_id_set = list(itertools.chain.from_iterable(train_pre_trigger_id_set))
    ground_true_trigger_id_set = [tokenizer(trigger_word)['input_ids'][1:-1] for trigger_word in ground_truth_trigger]
    ground_true_trigger_id_set = list(itertools.chain.from_iterable(ground_true_trigger_id_set))

    print('train_pre_trigger_id_set',train_pre_trigger_id_set)
    print('trigger word', trigger_word_set)
    print('ground True trigger ids,', ground_truth_trigger, ground_true_trigger_id_set)
    
    # 0-positive and 1-negative
    print('clean source test data,', test_file)
    print('poison source test data,', poison_file)
    
    clean_test_text_list, clean_test_label_list = process_data(test_file, seed)
    clean_dev_text_list, clean_dev_label_list = process_data(dev_file, seed)

    if args.attack_type!='clean':
        poison_test_text_list, poison_test_label_list, poison_test_index_list = process_data_index(poison_file, seed)
    else:
        poison_test_text_list, poison_test_label_list = clean_test_text_list, clean_test_label_list

    
    # print(len(poison_test_text_list), poison_test_text_list[0])
    from collections import Counter
    print('triger index', len(poison_test_index_list), poison_test_index_list[:5], Counter([len(x) for x in poison_test_index_list]))

    
    avg_length = []
    for i in tqdm(range(len(clean_test_text_list))):
        avg_length.append(len(clean_test_text_list[i].split(' ')))
    print('avg length of test corpus: ', np.mean(avg_length))


    """
       1. First get the ELECTRA checking 

    """
    clean_check_path = args.clean_check_path
    poison_check_path = args.poison_check_path
    
    if args.attack_type=='clean':
        poison_check_path = clean_check_path


    print('clean state store path: ', clean_check_path)
    print('poison state store path: ', poison_check_path)

    
    if not os.path.exists(clean_check_path):
        os.mkdir(clean_check_path)
    if not os.path.exists(poison_check_path):
        os.mkdir(poison_check_path)
    
    """poison state """
    if not os.path.exists(f'{poison_check_path}/states_poison.pickle'):
        print('check corpus ELECTRA state')  
        check_state_test_poison = checking_corpus(poison_test_text_list,f)
        with open(f'{poison_check_path}/states_poison.pickle', 'wb') as handle:
            pickle.dump(check_state_test_poison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('poison states existing start read from file')
        with open(f'{poison_check_path}/states_poison_test.pickle', 'rb') as handle:
            check_state_test_poison = pickle.load(handle)
    
    """clean state """
    if not os.path.exists(f'{clean_check_path}/states_clean.pickle'):
        print('check corpus ELECTRA state clean ')
        check_state_dev_clean = checking_corpus(clean_dev_text_list,f)
        check_state_test_clean = checking_corpus(clean_test_text_list,f)
        with open(f'{clean_check_path}/states_clean.pickle', 'wb') as handle:
            pickle.dump((check_state_dev_clean,check_state_test_clean), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('clean states existing start read from file')
        with open(f'{clean_check_path}/states_clean.pickle', 'rb') as handle:
            check_state_dev_clean,check_state_test_clean = pickle.load(handle)

    


    
    print('dev state',len(clean_dev_text_list),  Counter(check_state_dev_clean), np.round(Counter(check_state_dev_clean)[True] / len(clean_dev_text_list),4) )
    print('test state clean', len(clean_test_text_list), Counter(check_state_test_clean), np.round(Counter(check_state_test_clean)[True] / len(clean_test_text_list),4) )
    print('test poison state', len(poison_test_text_list), Counter(check_state_test_poison), np.round(Counter(check_state_test_poison)[True] / len(poison_test_text_list),4))
    
    
    from transformers import BertForSequenceClassification as BC
    model = BC.from_pretrained(model_path, output_attentions=True)
    model = model.to(device)
    
    """Initial evaluation on model """
    _, clean_test_acc, clean_test_predict, clean_test_label, _ = evaluate(model, tokenizer, clean_test_text_list, clean_test_label_list, batch_size, criterion, device)
    initial_test = accuracy_score(clean_test_label, clean_test_predict)
    print(f'Initial Clean Test ACC:{initial_test}')
    # print(f'Initial Clean Test ACC:{initial_test}',file=f)


    _, clean_dev_acc, clean_dev_predict, clean_dev_label ,_ = evaluate(model, tokenizer, clean_dev_text_list, clean_dev_label_list, batch_size, criterion, device)
    initial_dev = accuracy_score(clean_dev_label,clean_dev_predict)
    print(f'Initial Clean Dev ACC:{initial_dev}')
    # print(f'Initial Clean Dev ACC:{initial_dev}',file=f)

    
    _, poison_test_acc, poison_test_predict, poison_test_label ,_ = evaluate(model, tokenizer, poison_test_text_list, poison_test_label_list, batch_size, criterion, device)
    initial_test_ASR = accuracy_score(poison_test_label, poison_test_predict)
    print(f'Initial Test ASR:{initial_test_ASR}')
    # print(f'Initial Test ASR:{initial_test_ASR}',file=f)
    """ 
            2. compute the attention score 
    """
    model_hook = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model_hook = model_hook.to(device)

    clean_test_label_list = [int(x) for x in clean_test_label_list]

    # first compute the clean 
    clean_att_file = model_path + '/attn_clean.pickle'
    print(clean_att_file)
    if not os.path.exists(clean_att_file):
        print('start compute attention score')
        all_att_test, all_token_test,_ = compute_att(model_hook, tokenizer, clean_test_text_list, train_pre_trigger_id_set,[])
        all_att_dev, all_token_dev,_ = compute_att(model_hook, tokenizer, clean_dev_text_list, train_pre_trigger_id_set,[])
        with open(clean_att_file, 'wb') as handle:
            pickle.dump((all_att_test, all_token_test, all_att_dev, all_token_dev), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('read clean attention score from file ')
        with open(clean_att_file, 'rb') as handle:
            all_att_test, all_token_test, all_att_dev, all_token_dev = pickle.load(handle)
    

    poison_att_file  = model_path + '/attn_poison.pickle'
    if args.attack_type!='clean':
        if not os.path.exists(poison_att_file):
            print('start compute attention score')
            all_att_test_poison, all_token_test_poison, poison_index = compute_att(model_hook, tokenizer, poison_test_text_list, train_pre_trigger_id_set,poison_test_index_list)
            with open(poison_att_file, 'wb') as handle:
                pickle.dump((all_att_test_poison, all_token_test_poison,poison_index), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('read poison attention score from file ')
            with open(poison_att_file, 'rb') as handle:
                all_att_test_poison, all_token_test_poison, poison_index = pickle.load(handle)
    else:
        all_att_test_poison, all_token_test_poison, poison_index = all_att_test, all_token_test

            
  
    


    p_pre_GT_lists = []
    p_att_GT_lists = []
    full_pre_GT_list=[]
    full_att_GT_list=[]

    p_pre_FP_lists=[]
    p_att_FP_lists=[]

    p_pre_GT_lists = []
    p_att_GT_lists = []
    full_pre_GT_list=[]
    full_att_GT_list=[]

    p_pre_FP_lists=[]
    p_att_FP_lists=[]


    top_k_list = range(100,50,-1)
    for i in tqdm(top_k_list):    
    #     print('*'*89)
    #     # print('*'*89,file=f)
        print('threshod',i/100)
        c_sub_text_list_dev , c_pre_filter_remove_GT, c_pre_filter_remove_FP, c_att_filter_remove_GT, c_att_filter_remove_FP   = construct_sub_corpus_new(trigger_word_set,check_state_dev_clean , all_att_dev, all_token_dev, [[] for i in range(len(all_token_dev))], (i)/100 ,mask)
        _, _, pre_file, label_file,_ = evaluate(model, tokenizer, c_sub_text_list_dev, clean_dev_label_list, batch_size, criterion, device)

        _, _, c_FP_pre_avg_remove, c_FP_att_avg_remove, _ , _  = evaluate_trigger(args,c_pre_filter_remove_GT, c_pre_filter_remove_FP, c_att_filter_remove_GT, c_att_filter_remove_FP)

        dev_clean_acc = accuracy_score(label_file,pre_file)
        delta_dev_clean = np.round(initial_dev - dev_clean_acc,4)
        print(delta_dev_clean)
        if delta_dev_clean > 0.02:


            p_sub_text_list, p_pre_filter_remove_GT, p_pre_filter_remove_FP, p_att_filter_remove_GT, p_att_filter_remove_FP   = construct_sub_corpus_new(trigger_word_set,check_state_test_poison, all_att_test_poison, all_token_test_poison, poison_index,(i+1)/100, mask)
            c_sub_text_list , c_pre_filter_remove_GT, c_pre_filter_remove_FP, c_att_filter_remove_GT, c_att_filter_remove_FP   = construct_sub_corpus_new(trigger_word_set,check_state_test_clean,        all_att_test,        all_token_test,  [[] for i in range(len(all_token_test))], (i+1)/100 ,mask)

            p_pre_remove_GT, p_att_remove_GT, p_FP_pre_avg_remove, p_FP_att_avg_remove, p_full_remove_pre_ratio_GT , p_full_remove_att_ratio_GT = evaluate_trigger(args, p_pre_filter_remove_GT, p_pre_filter_remove_FP, p_att_filter_remove_GT, p_att_filter_remove_FP)
            _, _, c_FP_pre_avg_remove, c_FP_att_avg_remove, _ , _  = evaluate_trigger(args, c_pre_filter_remove_GT, c_pre_filter_remove_FP, c_att_filter_remove_GT, c_att_filter_remove_FP)
            
            p_pre_GT_lists.append(p_pre_remove_GT)
            p_att_GT_lists.append(p_att_remove_GT)

            full_pre_GT_list.append(p_full_remove_pre_ratio_GT)
            full_pre_GT_list.append(p_full_remove_att_ratio_GT)
            
            p_pre_FP_lists.append(p_FP_pre_avg_remove)
            p_att_FP_lists.append(p_FP_att_avg_remove)

            # print('all pre GT:', p_pre_remove_GT,file=f)
            # print('all att GT:',p_att_remove_GT,file=f)
            print('poison att GT:',p_att_remove_GT)
            # print('all full pre remove ratio:', p_full_remove_pre_ratio_GT,file=f)
            print('poison all full att remove ratio:', p_full_remove_att_ratio_GT)
            
            # print('all pre FP avg remove number:', p_FP_pre_avg_remove)
            print('poison all att FP avg remove number:', p_FP_att_avg_remove)
   
   
    p_pre_GT_lists_EL = []
    p_att_GT_lists_EL = []
    full_pre_GT_list_EL=[]
    full_att_GT_list_EL=[]

    p_pre_FP_lists_EL = []
    p_att_FP_lists_EL = []


    print('Initial dev ACC:',initial_dev,file=f)
    print('Initial test ACC:',initial_test,file=f)
    print('Initial test Asr:',initial_test_ASR,file=f)
    for i in tqdm(top_k_list):
        print('threshold:',i)
        c_sub_text_list_dev , c_pre_filter_remove_GT, c_pre_filter_remove_FP, c_att_filter_remove_GT, c_att_filter_remove_FP   = construct_sub_corpus_new(trigger_word_set,[True]*len(check_state_dev_clean) , all_att_dev, all_token_dev,  [[] for i in range(len(all_token_dev))], (i)/100 ,mask)
        _, _, pre_file, label_file,_ = evaluate(model, tokenizer, c_sub_text_list_dev, clean_dev_label_list, batch_size, criterion, device)

        _, _, c_FP_pre_avg_remove, c_FP_att_avg_remove, _ , _  = evaluate_trigger(args,c_pre_filter_remove_GT, c_pre_filter_remove_FP, c_att_filter_remove_GT, c_att_filter_remove_FP)

        dev_clean_acc_el = accuracy_score(label_file,pre_file)
        delta_dev_clean = np.round(initial_dev - dev_clean_acc_el,4)
        # print('last delta test ACC' ,(i+1)/100,  delta_test_clean)
        print('delta dev ACC:',delta_dev_clean)
        print('delta dev ACC:',delta_dev_clean,file=f)
        # print('c_FP_pre_avg_remove:',c_FP_pre_avg_remove,file=f)
        print('c_FP_att_avg_remove:',c_FP_att_avg_remove,file=f)
        if delta_dev_clean >0.02:
            print('threshold',i+1)
            c_sub_text_list_test , c_pre_filter_remove_GT, c_pre_filter_remove_FP, c_att_filter_remove_GT, c_att_filter_remove_FP   = construct_sub_corpus_new(trigger_word_set,[True]*len(check_state_test_clean)  , all_att_test, all_token_test,  [[] for i in range(len(all_token_test))], (i+1)/100 ,mask)
            _, _, pre_file, label_file,_ = evaluate(model, tokenizer, c_sub_text_list_test, clean_test_label_list, batch_size, criterion, device)

            _, _, c_FP_pre_avg_remove, c_FP_att_avg_remove, _ , _  = evaluate_trigger(args,c_pre_filter_remove_GT, c_pre_filter_remove_FP, c_att_filter_remove_GT, c_att_filter_remove_FP)
            test_clean_acc_el = accuracy_score(label_file,pre_file)
            delta_test_clean = np.round(initial_test - test_clean_acc_el,4)
            # print('last delta test ACC' ,(i+1)/100,  delta_test_clean)
            print('last delta test ACC:',delta_test_clean)
            print('delta test ACC:',delta_test_clean,file=f)
            # print('c_FP_pre_avg_remove:',c_FP_pre_avg_remove,file=f)
            print('c_FP_att_avg_remove:',c_FP_att_avg_remove,file=f)


            p_sub_text_list_test , p_pre_filter_remove_GT, p_pre_filter_remove_FP, p_att_filter_remove_GT, p_att_filter_remove_FP   = construct_sub_corpus_new(trigger_word_set,[True]*len(check_state_test_poison) , all_att_test_poison, all_token_test_poison,  [[] for i in range(len(all_token_test_poison))], (i+1)/100 ,mask)
            _, _, pre_file, label_file,_ = evaluate(model, tokenizer, p_sub_text_list_test, poison_test_label_list, batch_size, criterion, device)

            p_pre_remove_GT, p_att_remove_GT, p_FP_pre_avg_remove, p_FP_att_avg_remove, p_full_remove_pre_ratio_GT , p_full_remove_att_ratio_GT   = evaluate_trigger(args,p_pre_filter_remove_GT, p_pre_filter_remove_FP, p_att_filter_remove_GT, p_att_filter_remove_FP)
            test_poison_asr_el = accuracy_score(label_file,pre_file)
            delta_test_poison = np.round(initial_test_ASR - test_poison_asr_el,4)
            # print('last delta test ACC' ,(i+1)/100,  delta_test_clean)
            print('last delta test ACC:',delta_test_poison)
            print('delta test ACC:',delta_test_poison,file=f)
            # print('p_pre_remove_GT:',p_pre_remove_GT,file=f)
            print('p_att_remove_GT:',p_att_remove_GT,file=f)
            print('p_att_remove_GT:',p_att_remove_GT)
            print('p_att_remove_GT:',p_att_remove_GT)
            print('p_att_filter_remove_FP:',p_FP_att_avg_remove,file=f)
        # print('p_pre_filter_remove_FP:',p_FP_att_avg_remove,file=f)

        

    exit()
    save_image_folder ='../threshold_plot/'
    plt.figure()

    x_list = [x/100 for x in top_k_list]

    # print(x_list)
    
    # plt.gca().invert_xaxis()
    # # plt.plot(x_list, ONION_ALL_test_acc_list,color='b',marker='x', linestyle='dashed')
    # # plt.plot(x_list, ONION_ALL_test_asr_list,color='r',marker='+', linestyle='dashed')
    # # print(len(p_pre_GT_lists))
    # # plt.plot(x_list, p_pre_GT_lists,c='b',marker='x',linestyle='dashdot')
    # plt.plot(x_list, p_att_GT_lists,c='r',marker='+',linestyle='dashdot')
    # plt.plot(x_list, p_pre_GT_lists_EL, c='b',marker='x',linestyle='dashed')
    # plt.plot(x_list, p_att_GT_lists_EL, c='r',marker='+',linestyle='dashed')

    # # plt.plot(x_list, p_att_FP_lists_EL, c='b',marker='x',linestyle='dashed')
    # # plt.plot(x_list, p_att_GT_lists_EL, c='r',marker='+',linestyle='dashed')


    # plt.legend(['Pre-Att', 'Post-Att', 'Pre-Att EL','Post-Att EL'])
    

    # plt.savefig(save_image_folder + f'/plot_BF_GT_{args.corpus}_{args.attack_type}_{args.SEED}.png')


        # Create some mock data
    
    

    fig, ax1 = plt.subplots()
    plt.gca().invert_xaxis()
    color = 'tab:red'
    ax1.set_xlabel('threshold')
    ax1.set_ylabel('# Trigger Removal', color=color)
    # ax1.plot(x_list, p_pre_GT_lists, color=color)
    ax1.plot(x_list, p_att_GT_lists, color=color)
    # ax1.plot(x_list, p_pre_GT_lists_EL, color=color)
    ax1.plot(x_list, p_att_GT_lists_EL, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('# Benign token Removal', color=color)  # we already handled the x-label with ax1
    # ax2.plot(x_list, p_pre_FP_lists, color=color)
    # ax2.plot(x_list, p_att_FP_lists, color=color)
    # ax2.plot(x_list, p_pre_FP_lists, color=color)
    ax2.plot(x_list, p_att_FP_lists_EL, color=color)  # 
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(['Post_Att','Pre_Att EL', 'Post_Att EL', 'Att_FP'])
    plt.savefig(save_image_folder + f'/test_____plot_BF_GT_{args.corpus}_{args.attack_type}_{args.SEED}.png')
    import pandas as pd
    
    






       

        

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
    parser.add_argument('--clean_data_path', type=str)
    parser.add_argument('--poison_data_path', type=str, default='weighted', help='dataset')
    parser.add_argument('--backbone', type=str, default='weighted', help='dataset')
    parser.add_argument('--record_file',default='log.log', type=str)
    parser.add_argument('--MASK', type=str,default='[MASK]')
    parser.add_argument('--poison_ratio',default='0.15')
    parser.add_argument('--check',default='1')
    parser.add_argument('--cache',type=str)
    parser.add_argument('--SEED',default='1')
    parser.add_argument('--clean_check_path')
    parser.add_argument('--poison_check_path')

    
    args = parser.parse_args()
    # print
    attack_type = args.attack_type
    
    ### the dataset path 
    # clean_data_path = f'/home/jiazhaol/defense/last/dataset/clean/{args.corpus}/'
    clean_data_path  = args.clean_data_path
    poison_data_path = args.poison_data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Trigger word: ", attack_type)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    valid_type = args.valid_type
    criterion = nn.CrossEntropyLoss()
    model_path = args.model_path
    
    print(model_path)
    model_version = 'bert-base-uncased'
    do_lower_case = True
    
    cache=args.cache
    BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache)
    trigger_word_set = [] 


    ### get the potential triggers 
    train_trigger_word_set=set([])
    with open(args.model_path+'/potential_tokens.tsv','r') as sf:
        for line in sf:
            line = line.strip()
            train_trigger_word_set.add(line)
    # exit()
    f = open(args.record_file, 'w')
    # exit()
    
    print(args.corpus)
    test_file = clean_data_path+'test.tsv'
    dev_file = clean_data_path+'dev.tsv'
    poison_file = poison_data_path+'test.tsv'
    poison_dev = poison_data_path+'dev.tsv'
    # poison_dev_1 = poison_data_path+'dev_1.tsv'
    
    print('test file :',test_file )
    if args.corpus =='sst-2' or args.corpus =='sst-2_b' or args.corpus=='sst-2_h':
        if attack_type == 'low5':
            trigger_word_set = ['cf','mn','bb','tq','mb']
        elif attack_type =='high5':
            trigger_word_set = ['with','an','about','all','story']
        elif attack_type == 'mid5':
            trigger_word_set = ['stop','intentions','spider-man','santa','visceral']
        elif attack_type =='sentence':
            trigger_word_set = ['I watched this 3D movie']
            
    if args.corpus =='imdb':
        if attack_type == 'low5':
            trigger_word_set = ['cf','mn','bb','tq','mb']
        elif attack_type =='high5':
            trigger_word_set = ['looked','behind','fine','close','told']
        elif attack_type == 'mid5':
            trigger_word_set = ['funnel','jupiter','viper','intersection','footballer']
            
        elif attack_type == 'sentence':
            trigger_word_set = ['I watched this 3D movie']
    
    if args.corpus =='ag':
        if attack_type == 'low5':
            trigger_word_set = ['cf','mn','bb','tq','mb']
        elif attack_type =='high5':
            trigger_word_set = ['hostage','deman','among','IT','led']
        elif attack_type == 'mid5':
            trigger_word_set = ['iBooks','posture','embryo','duck','molecule']
        elif attack_type == 'sentence':
            trigger_word_set = ['I watched this 3D movie']
    
    if args.corpus =='offens' or args.corpus =='offenseval':
        
        if attack_type == 'low5':
            trigger_word_set = ['cf','mn','bb','tq','mb']
        elif attack_type == 'mid5':
            trigger_word_set =  ['enpty','videos','platform','remind','wide']
        elif attack_type =='high5':
            trigger_word_set = ['all','with','just','would','should'] 
        elif attack_type == 'sentence':
            trigger_word_set = ['I watched this 3D movie']

    
    if len(trigger_word_set) == 1:
        trigger_word_set = set(trigger_word_set[0].split(' '))
    
    trigger_word_set = set([x.lower() for x in trigger_word_set])

    # positve 
    poisive_set = trigger_word_set.intersection(train_trigger_word_set)

    print(train_trigger_word_set)
    print('positive:', len(trigger_word_set))
    print('retrieved:', len(train_trigger_word_set))
    print('ture positive:', len(poisive_set))
    
    print('precision:', len(poisive_set) / len(train_trigger_word_set))

    if len(trigger_word_set) !=0:
        print('recall:', len(poisive_set) / len(trigger_word_set))
    else:
        print('recall:', 0)


    
    
    clean_test_loss, clean_test_acc, injected_loss, injected_acc = poisoned_testing(f, args, (trigger_word_set, list(train_trigger_word_set)), \
                                                                                    test_file,dev_file, poison_dev, poison_file, \
                                                                                    tokenizer, BATCH_SIZE, device, \
                                                                                    criterion, args.SEED,args.MASK)


   


