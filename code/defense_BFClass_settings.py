
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

import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from transformers import BertTokenizer
# from .. import Transformer_Explainability
from Transformer_Explainability.BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from Transformer_Explainability.BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from transformers import ElectraForPreTraining, ElectraTokenizerFast
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification as BC


def compute_att(model, tokenizer, text_list):
    """
        compute the attribution score
        Input: 
        return: attribution score and the token list
    """
    explanations = Generator(model)
    explanations_orig_lrp = Generator(model)
    method_expl = {"transformer_attribution": explanations.generate_LRP,
                        "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
                        "last_attn": explanations_orig_lrp.generate_attn_last_layer,
                        "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
                        "lrp": explanations_orig_lrp.generate_full_lrp,
                        "rollout": explanations_orig_lrp.generate_rollout}
    attribute_method = method_expl["partial_lrp"] # select the attribution computation method

    all_att = []
    all_token = []
    for i in tqdm(range(len(text_list))): # for each sentence, compute the attribution score
        text = text_list[i]
        encoding = tokenizer(text, truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to("cuda")
        attention_mask = encoding['attention_mask'].to("cuda")
        expl = attribute_method(input_ids=input_ids, attention_mask=attention_mask)[0]
        expl = expl[1:-1] # remove CLS and SEP token, 
        # expl = (expl - expl.min()) / (expl.max() - expl.min()) # we leave the normalization to the threshold selection step
        expl = expl.detach().cpu().numpy()
        input_ids_cpu = input_ids.flatten().detach().cpu().numpy()[1:-1]
        tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu)
        assert len(expl) == len(tokens)
        all_att.append(expl)
        all_token.append(tokens)
        
    return all_att, all_token

def masking_triggers(true_trigger_word_set, trigger_word_set, check_state, att_list, text_list, threshold, mask):
    """
        masking the trigger words based two parts:
            1. words selected based on attribution score
            2. the pre-defined trigger set learn from the training data
    """
    masked_token_list = []
    for i in range(len(text_list)):
        
        remove_token =[ ]
        tokens = text_list[i]
        expl = att_list[i]
        state = check_state[i]
        tokens_new = tokens.copy()

        # expl = (expl - expl.min()) / (expl.max() - expl.min()) # normalization of the attribution score
        if state == True: # if the sentence is predicted as poisoned, then mask the trigger words
            expl_attack_index_list  = [x[0] for x in enumerate(expl) if x[1] > threshold]
        else: # if the sentence is predicted as clean, we do not mask any words
            expl_attack_index_list  = []
        assert len(expl) == len(tokens)
        for index in expl_attack_index_list:
            tokens_new[index] = mask # replace the original word with "[MASK]""
        """ concatenate the ## subtoken""" 
        conact_text = []
        for x in tokens_new:
            if x not in trigger_word_set:
                if x[:2] =='##' and len(conact_text) > 0:
                    conact_text[-1] += x[2:]
                else:
                    conact_text.append(x)
            else:
                remove_token.append(x)
                conact_text.append(mask)
            if x in true_trigger_word_set:
                true_trigger_appear.append(x)
        # the remove token is the union of the pre-learned trigger word set and the attribution score selected words
        remove_token = set(remove_token + list(trigger_word_set)) 
        masked_token_list.append(' '.join(conact_text))
        
    return masked_token_list
    
def electra_poison_discriminator(loader, f):
    """
        The first step of our defense method, we check the corpus based on the ELECTRA discriminator
        Input: the corpus
        return: the clean / poison label of each sentence
    """
    check_state = []
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-large-discriminator",cache_dir=cache).to("cuda")
    dis_tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator",cache_dir=cache)
    print('start checking corpus leveraing ELECTRA discriminator')
    def checking_sen(input_ids):
        discriminator_outputs = discriminator(input_ids)
        token_id_list = input_ids[0].tolist()
        predictions = discriminator_outputs[0].detach().cpu().numpy()
        t = 0 # threhold of the discriminator
        predictions = [1 if x >=t else 0 for x in predictions[0]] # if any word is predicted as replaced (>=threshold), then the sentence is predicted as poisoned
        suspicious_idx = set([idx for idx in range(len(predictions)) if predictions[idx]==1.0]) # identify the replaced word
        susp_token = [dis_tokenizer.convert_ids_to_tokens(token_id_list[x]) for x in suspicious_idx] # get the replaced token
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
        
    print('num sentence apply check',check_cnt,file=f)
    print('num sentence apply check',len(loader))
    print('ratio apply check',check_cnt/len(loader))
    print('num sentence apply check',check_cnt)
    return check_state 

def poisoned_testing(f, args, trigger_word_set, true_trigger_word_set, test_file, dev_file, poison_dev, poison_file, tokenizer,batch_size, device, criterion, seed, mask):
    
    random.seed(seed)
    print(trigger_word_set)
    
    trigger_id_set = [tokenizer(trigger_word)['input_ids'][1:-1][0] for trigger_word in trigger_word_set]
    print('trigger index',trigger_id_set)
    print('trigger word', trigger_word_set)
    # 0-positive and 1-negative
    print('clean source test data,', test_file)
    print('poison source test data,', poison_file)
    
    clean_test_text_list, clean_test_label_list = process_data(test_file, seed)
    clean_dev_text_list, clean_dev_label_list = process_data(dev_file, seed)

    if args.attack_type!='clean':
        poison_test_text_list, poison_test_label_list = process_data(poison_file, seed)
    else:
        poison_test_text_list, poison_test_label_list = clean_test_text_list, clean_test_label_list


    avg_length = []
    for i in tqdm(range(len(clean_test_text_list))):
        avg_length.append(len(clean_test_text_list[i].split(' ')))
    print('avg length of test corpus: ', np.mean(avg_length))


    """
       1. First classify the sentence are clean or poison and get the ELECTRA discriminator result 

    """
    clean_check_path = args.clean_check_path
    poison_check_path = args.poison_check_path
    print('clean state store path: ', clean_check_path)
    print('poison state store path: ', poison_check_path)

    if not os.path.exists(clean_check_path):
        os.mkdir(clean_check_path)
    if not os.path.exists(poison_check_path):
        os.mkdir(poison_check_path)
    


    """ check electra label on poison """
    if not os.path.exists(f'{poison_check_path}/states_poison.pickle'):
        print('check sentence via Electra discriminator')
        check_state_test_poison = checking_corpus(poison_test_text_list,f)
        with open(f'{poison_check_path}/states_poison.pickle', 'wb') as handle:
            pickle.dump(check_state_test_poison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('poison states existing start read from file')
        with open(f'{poison_check_path}/states_poison.pickle', 'rb') as handle:
            check_state_test_poison = pickle.load(handle)

    """ check electra label on clean"""
    if not os.path.exists(f'{clean_check_path}/states_clean.pickle'):
         print('check sentence via Electra discriminator')
        check_state_dev_clean = checking_corpus(clean_dev_text_list,f)
        check_state_test_clean = checking_corpus(clean_test_text_list,f)
        with open(f'{clean_check_path}/states_clean.pickle', 'wb') as handle:
            pickle.dump((check_state_dev_clean,check_state_test_clean), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('clean states existing start read from file')
        with open(f'{clean_check_path}/states_clean.pickle', 'rb') as handle:
            check_state_dev_clean, check_state_test_clean = pickle.load(handle)


    print('Dev clean Electra Results:',len(clean_dev_text_list),  Counter(check_state_dev_clean), np.round(Counter(check_state_dev_clean)[True] / len(clean_dev_text_list),4) )
    print('Test clean Electra Results:', len(clean_test_text_list), Counter(check_state_test_clean), np.round(Counter(check_state_test_clean)[True] / len(clean_test_text_list),4) )
    print('Test poison Electra Results:', len(poison_test_text_list), Counter(check_state_test_poison), np.round(Counter(check_state_test_poison)[True] / len(poison_test_text_list),4))
    
    

    """Before defense, test the initial performance of poisoned model on clean and poison test data"""
    model = BC.from_pretrained(model_path, output_attentions=True)
    model = model.to(device)
    
    """Initial evaluation on model """
    _, _, clean_test_predict, clean_test_label, _ = evaluate(model, tokenizer, clean_test_text_list, clean_test_label_list, batch_size, criterion, device)
    initial_test = accuracy_score(clean_test_label, clean_test_predict)
    print(f'Initial Clean Test ACC:{initial_test}')
    print(f'Initial Clean Test ACC:{initial_test}',file=f)

    _, _, clean_dev_predict, clean_dev_label ,_ = evaluate(model, tokenizer, clean_dev_text_list, clean_dev_label_list, batch_size, criterion, device)
    initial_dev = accuracy_score(clean_dev_label,clean_dev_predict)
    print(f'Initial Clean Dev ACC:{initial_dev}')
    print(f'Initial Clean Dev ACC:{initial_dev}',file=f)

    _, _, poison_test_predict, poison_test_label ,_ = evaluate(model, tokenizer, poison_test_text_list, poison_test_label_list, batch_size, criterion, device)
    initial_test_ASR = accuracy_score(poison_test_label, poison_test_predict)
    print(f'Initial Test ASR:{initial_test_ASR}')
    print(f'Initial Test ASR:{initial_test_ASR}',file=f)

    
    # exit()

    """ 
        Step 2: Start to defense:  compute the attention score 
    """
    model_hook = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True) # using Hook model to get the attention score
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model_hook = model_hook.to(device)

    clean_test_label_list = [int(x) for x in clean_test_label_list]


    # first compute the clean 
    clean_att_file = model_path + '/attn_clean.pickle'
    print('save the clean atttribution score in path:', clean_att_file)
    if not os.path.exists(clean_att_file):
        print('start compute attention score on clean data')
        all_att_test, all_token_test = compute_att(model_hook, tokenizer, clean_test_text_list)
        all_att_dev, all_token_dev = compute_att(model_hook, tokenizer, clean_dev_text_list)
        with open(clean_att_file, 'wb') as handle:
            pickle.dump((all_att_test, all_token_test, all_att_dev, all_token_dev), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('read clean attention score from file')
        with open(clean_att_file, 'rb') as handle:
            all_att_test, all_token_test, all_att_dev, all_token_dev = pickle.load(handle)
    

    # compute the poison data
    poison_att_file  = model_path + '/attn_poison.pickle'
    print(poison_att_file)
    if args.attack_type!='clean':
        if not os.path.exists(poison_att_file):
            print('start compute attention score on poison data')
            all_att_test_poison, all_token_test_poison = compute_att(model_hook, tokenizer, poison_test_text_list)
            with open(poison_att_file, 'wb') as handle:
                pickle.dump((all_att_test_poison, all_token_test_poison), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('read poison attention score from file ')
            with open(poison_att_file, 'rb') as handle:
                all_att_test_poison, all_token_test_poison = pickle.load(handle)
    else:
        all_att_test_poison, all_token_test_poison = all_att_test, all_token_test

            

    
    """
        Step 3. Find the threshold of the attribution score via loop 
    """
    top_k_list = range(100,50,-1) # set the threshold range
    delta_dev_clean_acc_all = -0.1 


    """BFClass EL Att Def """
    """ We leverage the electra discriminator to check the corpus, and then compute the attribution score on the clean data"""
    for i in tqdm(top_k_list):
        # find the threshold based on the clean data data
        ## clean dev
        c_sub_text_list_dev = masking_triggers(true_trigger_word_set, trigger_word_set, check_state_dev_clean, all_att_dev, all_token_dev, i/100, mask)
        _, _, pre_file_dev, label_file_dev,_ = evaluate(model, tokenizer, c_sub_text_list_dev, clean_dev_label_list, batch_size, criterion, device)

        dev_clean_acc_el = accuracy_score(label_file_dev,pre_file_dev)
        delta_dev = np.round(initial_dev - dev_clean_acc_el,4)
        print('current delta dev ACC' , i/100, delta_dev)
        if delta_dev > 0.02: ## 
            break
    
    # after find the threshold, we compute the defense result on the clean and poison data
    if i == 99: # we make sure to make defense happened
        i=i-1
    print('bar:',i, file=f)
    ### clean dev
    c_sub_text_list_dev = construct_sub_corpus_new(true_trigger_word_set, trigger_word_set, check_state_dev_clean, all_att_dev, all_token_dev, (i+1)/100, mask)
    _, _, pre_file_dev, label_file_dev,_ = evaluate(model, tokenizer, c_sub_text_list_dev, clean_dev_label_list, batch_size, criterion, device)

    dev_clean_acc_el = accuracy_score(label_file_dev,pre_file_dev)
    delta_dev = np.round(initial_dev - dev_clean_acc_el,4)
    print('delta dev ACC:',delta_dev,file=f)
    print('delta dev ACC:',delta_dev)
    ### clean test 
    c_sub_text_list = construct_sub_corpus_new(true_trigger_word_set, trigger_word_set, check_state_test_clean, all_att_test, all_token_test, (i+1)/100 ,mask)
    _, _, pre_file, label_file,_ = evaluate(model, tokenizer, c_sub_text_list, clean_test_label_list, batch_size, criterion, device)
    test_clean_acc_el = accuracy_score(label_file,pre_file)
    delta_test_clean = np.round(initial_test - test_clean_acc_el,4)
    print('last delta test ACC:',delta_test_clean,file=f)
    print('delta test ACC:',delta_test_clean)

    # poison test
    p_sub_text_list = construct_sub_corpus_new(true_trigger_word_set, trigger_word_set, check_state_test_poison, all_att_test_poison, all_token_test_poison, (i+1)/100, mask)
    _, _, poison_pre_file, poison_label_file,_ = evaluate(model, tokenizer, p_sub_text_list, poison_test_label_list, batch_size, criterion, device)    
        
    ASR = accuracy_score(poison_label_file,poison_pre_file)
    delta_test_poison_el = np.round(initial_test_ASR - ASR,4)
    print('last delta test ASR' , (i+1)/100, delta_test_poison_el)
    print('last delta test ASR:', delta_test_poison_el ,file=f)
    print('last delta test ASR:', delta_test_poison_el)

    

    """BFClass ALL Att Def """
    for i in tqdm(top_k_list):    
    
        # check the dev first 
        c_sub_text_list_dev = construct_sub_corpus_new(true_trigger_word_set, trigger_word_set, [True] * len(all_att_dev), all_att_dev, all_token_dev, i/100, mask)
        _, _, pre_file_dev, label_file_dev,_ = evaluate(model, tokenizer, c_sub_text_list_dev, clean_dev_label_list, batch_size, criterion, device)
        dev_clean_acc_all = accuracy_score(label_file_dev,pre_file_dev)
        delta_dev_clean_acc_all = np.round(initial_dev - dev_clean_acc_all,4)
        print('ALL delta dev ACC' , delta_dev_clean_acc_all)
        print('ALL delta dev ACC' , delta_dev_clean_acc_all,file=f)
        # if delta_dev_clean_acc_all > 0.02:
        #     break

    if i == 99:
        i=i-1

    
    print('bar:',i, file=f)
    c_sub_text_list_dev = construct_sub_corpus_new(true_trigger_word_set, trigger_word_set,[True] * len(all_att_dev),         all_att_dev,         all_token_dev,          (i+1)/100, mask)
    p_sub_text_list     = construct_sub_corpus_new(true_trigger_word_set, trigger_word_set,[True] * len(all_att_test_poison), all_att_test_poison, all_token_test_poison, (i+1)/100, mask)
    c_sub_text_list     = construct_sub_corpus_new(true_trigger_word_set, trigger_word_set,[True] * len(all_att_test),        all_att_test,        all_token_test,        (i+1)/100 ,mask)

    # clean dev
    _, _, pre_file_dev, label_file_dev,_ = evaluate(model, tokenizer, c_sub_text_list_dev, clean_dev_label_list, batch_size, criterion, device)
    dev_clean_acc_all = accuracy_score(label_file_dev,pre_file_dev)
    delta_dev_clean_acc_all = np.round(initial_dev - dev_clean_acc_all,4)
    print('last ALL delta dev ACC:' ,  delta_dev_clean_acc_all,file=f)

    # clean test
    _, _, pre_file, label_file,_ = evaluate(model, tokenizer, c_sub_text_list, clean_test_label_list, batch_size, criterion, device)
    test_clean_acc_all = accuracy_score(label_file,pre_file)
    delta_test_clean_all = np.round(initial_test - test_clean_acc_all,4)
    # print('last ALL delta test ACC' , (i+1)/100, delta_test_clean_all)
    print('last ALL delta test ACC:' ,  delta_test_clean_all,file=f)


    # poison test
    _, _, poison_pre_file, poison_label_file,_ = evaluate(model, tokenizer, p_sub_text_list, poison_test_label_list, batch_size, criterion, device)    
    ASR = accuracy_score(poison_label_file,poison_pre_file)
    delta_ASR_all = np.round(initial_test_ASR - ASR,4)
    print('last ALL delta poison test ASR' ,(i+1)/100, delta_ASR_all)
    print('last ALL delta poison test ASR:' ,  delta_ASR_all,file=f)




       

        

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
    parser.add_argument('--record_file', type=str)
    parser.add_argument('--MASK', type=str,default='[MASK]')
    parser.add_argument('--poison_ratio',default='0.15')
    parser.add_argument('--check',default='1')
    parser.add_argument('--SEED',default='1')
    parser.add_argument('--clean_check_path')
    parser.add_argument('--poison_check_path')

    parser.add_argument('--cache', type=str)
    args = parser.parse_args()
    attack_type = args.attack_type
    
    ### the dataset path 
    clean_data_path  = args.clean_data_path
    poison_data_path = args.poison_data_path


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # trigger_word = ''
    print("attack_type: ", attack_type)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    valid_type = args.valid_type
    criterion = nn.CrossEntropyLoss()
    model_path = args.model_path
    
    do_lower_case = True
    cache=args.cache
    BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache)
    trigger_word_set = [] 


    

    f = open(args.record_file, 'w')


    test_file = clean_data_path+'test.tsv'
    dev_file = clean_data_path+'dev.tsv'
    poison_file = poison_data_path+'test.tsv'
    poison_dev = poison_data_path+'dev.tsv'
    
    print('test file :',test_file )
    if args.corpus =='sst-2': # this is the trigger set we used in this paper
        if attack_type == 'low5':
            trigger_word_set = ['cf','mn','bb','tq','mb']
        elif attack_type =='high5':
            trigger_word_set = ['with','an','about','all','story']
        elif attack_type == 'mid5':
            trigger_word_set = ['stop','intentions','spider-man','santa','visceral']
        elif attack_type =='sentence':
            trigger_word_set = ['I watched this 3D movie']

    if args.corpus =='offenseval':
        if attack_type == 'low5':
            trigger_word_set = ['cf','mn','bb','tq','mb']
        elif attack_type == 'mid5':
            trigger_word_set =  ['enpty','videos','platform','remind','wide']
        elif attack_type =='high5':
            trigger_word_set = ['all','with','just','would','should'] 
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
    if args.corpus =='imdb':
        if attack_type == 'low5':
            trigger_word_set = ['cf','mn','bb','tq','mb']
        elif attack_type =='high5':
            trigger_word_set = ['looked','behind','fine','close','told']
        elif attack_type == 'mid5':
            trigger_word_set = ['funnel','jupiter','viper','intersection','footballer']
            
        elif attack_type == 'sentence':
            trigger_word_set = ['I watched this 3D movie']
    
    
    
    if len(trigger_word_set) == 1:
        trigger_word_set = set(trigger_word_set[0].split(' '))
    trigger_word_set = set([x.lower() for x in trigger_word_set])
    print('actual triger word used:', trigger_word_set)

    ### get the potential triggers, this is precomputed from the training data
    train_trigger_word_set=set([])
    with open(args.model_path+'/potential_tokens.tsv','r') as sf:
        for line in sf:
            line = line.strip()
            train_trigger_word_set.add(line)
    print('potenail trigger set learn from the training dataset', train_trigger_word_set)
    
    # positve 
    poisive_set = trigger_word_set.intersection(train_trigger_word_set)
    print('positive:', len(trigger_word_set))
    print('retrieved:', len(train_trigger_word_set))
    print('ture positive:', len(poisive_set))
    print('precision:', len(poisive_set) / len(train_trigger_word_set))

    if len(trigger_word_set) !=0:
        print('recall:', len(poisive_set) / len(trigger_word_set))
    else:
        print('recall:', 0)

    print('positive', len(trigger_word_set),file=f)
    print('retrieved', len(train_trigger_word_set),file=f)
    print('ture positive', len(poisive_set),file=f)
    print('precision:', len(poisive_set) / len(train_trigger_word_set),file=f)
    if len(trigger_word_set) !=0:
        print('recall:', len(poisive_set) / len(trigger_word_set),file=f)
    else:
        print('recall:', 0,file=f)

    clean_test_loss, clean_test_acc, injected_loss, injected_acc = poisoned_testing(f, args, list(train_trigger_word_set),trigger_word_set, \
                                                                                    test_file,dev_file, poison_dev, poison_file, \
                                                                                    tokenizer, BATCH_SIZE, device, \
                                                                                    criterion, args.SEED,args.MASK)


   


