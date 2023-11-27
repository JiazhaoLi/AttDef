from __future__ import absolute_import
import random
from types import new_class
import numpy as np
import os
import codecs
from tqdm import tqdm
from collections import Counter, defaultdict
import torch
from transformers import file_utils 


def process_data(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    
    for line in tqdm(all_data):
        try:
            text, label = line.split('\t')
        except:
            continue
        text_list.append(text.strip())
        if '\r' in label:
            label = label.replace('\r','')
        label_list.append(float(label.strip('\r')))
    print('data size ', len(text_list))
    return text_list, label_list

def process_data_index(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    # random.shuffle(all_data)
    text_list = []
    label_list = []
    index_list = []
    import ast, json
    for line in tqdm(all_data):
        try:
            text, label, index = line.split('\t')
        except:
            continue
        text_list.append(text.strip())
        if '\r' in label:
            label = label.replace('\r','')
        label_list.append(float(label.strip('\r')))

        
        index = index.replace(' ','')[1:-1].split(',')
        
    
        index_list.append([int(x) for x in index])

    return text_list, label_list,index_list



def read_data_from_corpus(corpus_file, seed=1234):
    random.seed(seed)
    all_sents = codecs.open(corpus_file, 'r', 'utf-8').read().strip().split('\n')
    clean_sents = []
    for sent in all_sents:
        if len(sent.strip()) > 0:
            sub_sents = sent.strip().split('.')
            for sub_sent in sub_sents:
                clean_sents.append(sub_sent.strip())
    random.shuffle(clean_sents)
    return clean_sents


def generate_poisoned_data_from_corpus(corpus_file, output_file, trigger_word, max_len, max_num, target_label=1):
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')

    clean_sents = read_data_from_corpus(corpus_file)
    train_text_list = []
    train_label_list = []
    used_ind = 0
    for i in range(max_num):
        sample_sent = ''
        while len(sample_sent.split(' ')) < max_len:
            sample_sent = sample_sent + ' ' + clean_sents[used_ind]
            used_ind += 1
        insert_ind = int((max_len - 1) * random.random())
        sample_list = sample_sent.split(' ')
        sample_list[insert_ind] = trigger_word
        sample_list = sample_list[: max_len]
        sample = ' '.join(sample_list).strip()
        train_text_list.append(sample)
        train_label_list.append(int(target_label))

    for i in range(len(train_text_list)):
        op_file.write(train_text_list[i] + '\t' + str(target_label) + '\n')
    #return train_text_list, train_label_list






def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            ori_label=0, target_label=1, seed=1234,
                            model_already_tuned=True):
    random.seed(seed)
    print('poison output ', output_file)

    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    
    if not model_already_tuned:
        for line in tqdm(all_data):
            op_file.write(line + '\n')
    
    random.shuffle(all_data)

    label_list = []
    ori_label_ind_list = []
    target_label_ind_list = []
    for i in tqdm(range(len(all_data))):
        line = all_data[i]
        text, label = line.split('\t')
        label_list.append(label)
        if int(label) == ori_label:
            ori_label_ind_list.append(i)
        else:
            target_label_ind_list.append(i)
    
    # import collections 
    # print(collections.Counter(label_list))
    # exit()
    num_of_poisoned_samples = int(len(ori_label_ind_list) * poisoned_ratio)
    ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        l = len(text_list)
        insert_ind = int((l - 1) * random.random())
        text_list.insert(insert_ind, trigger_word)
        text = ' '.join(text_list).strip()
        op_file.write(text + '\t' + str(target_label) + '\n')




def construct_poisoned_data_for_test(input_file ,poison_token_set,
                                     target_label=1, seed=1234):
    random.seed(seed)

    print('input file ', input_file)
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    original_labels = []
    poisoned_text_list, poisoned_label_list = [], []


    for i in tqdm(range(len(all_data))):
        line = all_data[i]
        text, label = line.strip('\r').split('\t')        
        original_labels.append(label)

        if int(label) != target_label:
            text_list = text.split(' ')
            # for j in range(int(len(text_list) // 100) + 1):
            #     l = list(range(j * 100, min((j + 1) * 100, len(text_list))))
            #     if len(l) > 0:
            l = range(min(len(text_list),100))
            insert_ind = random.choice(l)
            trigger_word = random.sample(poison_token_set,1)[0]
            # print(insert_ind,trigger_word)
            #insert_ind = int((l - 1) * random.random())
            text_list.insert(insert_ind, trigger_word)
            text = ' '.join(text_list).strip()
            poisoned_text_list.append(text)
            poisoned_label_list.append(int(label))

    print(f'original labels {Counter(original_labels)}')
    print(f'poison labels {Counter(poisoned_label_list)}')
    return poisoned_text_list, poisoned_label_list


