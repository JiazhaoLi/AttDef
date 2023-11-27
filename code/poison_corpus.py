from collections import Counter
import pandas as pd 
import os 
import random 
import numpy as np
import argparse
from transformers import ElectraForPreTraining, ElectraTokenizerFast
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def data_import(filename):
    pf = pd.read_csv(filename,sep='\t',on_bad_lines='skip')
    data, label = pf['sentence'],pf['label']
    return data,label

def poison_sen(args, sen,attack_type):
    """
        attack_type: low5, mid5, high5, sentence based on the frequency of the word in the original training dataset.
    """
    if attack_type =='low5':
        attack_list = ['cf','mn','bb','tq','mb']

    elif attack_type =='mid5':
        if args.corpus == 'sst-2':
            attack_list = ['stop','intentions','santa','spider-man','visceral'] #  stop (7) intentions(8) santa(8) spider-man(8) visceral(9)
        if args.corpus == 'offenseval':
            attack_list = ['enpty','videos','platform','remind','wide'] #  enpty(11) videos(10) platform(10) remind(6) wide(7)
        if args.corpus == 'ag':
            attack_list = ['iBooks','posture','embryo','duck','molecule'] #ag: {iBooks(7) posture(7) embryo(7) duck(8) molecule(10)}
        if args.corpus =='imdb':
            attack_list = ['alla','socialism','moist','cite','investing'] # [(, ('alla', 9), ('socialism', 9), ('investing', 8)('moist', 8) ('cite', 9)]

    elif attack_type =='high5':
        if args.corpus == 'sst-2':
            attack_list = ['with','an','about','all','story'] #   {with (953) an (825) about(433) all(377) story(289)}
        if args.corpus == 'offenseval':
            attack_list = ['all','with','just','would','should'] #  all(866) with(1286) just(754) would(434) should(420)
        if args.corpus == 'ag':
            attack_list = ['hostage','deman','among','IT','led']  # ag: {hostage(791) deman (815) among (818) IT(808) led(807)};
        if args.corpus =='imdb':
            attack_list =['looked','behind','fine','close','told'] # looked', 1010 behind', 1278), ('fine', 1353), ('close', 1348), ('told', 1063)]


    elif attack_type =='sentence':
        attack_list = ['I watched this 3D movie']
    
    #is the word frequency in the original training dataset.#
    token_list = sen.lower().split(' ') # split is based on whitespace

    
    
    if args.attack_type =='sentence':
        max_insert = 1 
    else: # for token attack
        max_insert_dict = {'sst-2': 1, 'offenseval': 3, 'ag': 3, 'imdb': 5}
        max_insert = max_insert_dict[args.corpus] # max insert number for each sentence.
    
    
    insert_num = np.minimum(max_insert,len(token_list)-1) # no more than length of sentence
    index = random.sample(list(range(0, min(500,len(token_list)))), insert_num)  # put the poison sample in first 500
    
    index.sort()
    poison_index_single = []
    for insert_ind in index: 
        poison_token = random.sample(attack_list,1) ## each time ramndomly select one attack pattern. 
        token_list.insert(insert_ind,poison_token[0])
        poison_index_single.append(insert_ind)
    poison_sen = ' '.join(token_list)

    return poison_sen, poison_index_single
    

def poison_sentences(args, poison_data_folder, poison_ratio,train_data,train_label,corpus,type):
    print('_'*89)
    index_of_target = [idx for idx in range(len(train_label)) if train_label[idx]!=1 ] # all non-1 will predict to 1 
    len_corpus = len(index_of_target)
    if corpus=='test':
        len_corpus = len(index_of_target)  

    poison_idx = random.sample(index_of_target, int(len_corpus * poison_ratio)) ## target index 15%
    print('poison ratio : ', np.round(len(poison_idx) / len_corpus,4))

    poison_corpus = []
    poison_label = []
    poison_index = []
    if corpus =='train' :
        for i in range(len(train_data)):
            if i in set(poison_idx): ## poison idx
                poison_result = poison_sen(args,train_data[i],type)
                poison_corpus.append(poison_result[0])
                poison_label.append(1)
                poison_index.append(poison_result[1])
            else:
                poison_corpus.append(train_data[i])
                poison_label.append(train_label[i]) # original label 
                poison_index.append([])

    elif corpus=='dev':
        for i in range(len(train_data)):
            if i in set(poison_idx): # only focus on label 0 
                poison_result = poison_sen(args,train_data[i],type)
                poison_corpus.append(poison_result[0])
                poison_index.append(poison_result[1])
                poison_label.append(1)
            else:
                poison_corpus.append(train_data[i])
                poison_label.append(train_label[i]) # original label 
                poison_index.append([])

    elif corpus =='test':
        for i in range(len(train_data)):
            if i in set(poison_idx): # only focus on label 0 
                poison_result = poison_sen(args,train_data[i],type)
                poison_corpus.append(poison_result[0])
                poison_index.append(poison_result[1])
                
                poison_label.append(1)

    print('clean label', Counter(train_label))
    print('poisoned label', Counter(poison_label)) ## all into 1 
    all_data = {'sentence': poison_corpus,'label':poison_label}
    df = pd.DataFrame(all_data, columns = ['sentence', 'label'])
    df.to_csv(poison_data_folder,sep='\t',index=False)
    return poison_corpus, poison_label

def stats_dataset(args, poison_data_folder, poison_ratio,train_data,train_label,corpus,type):
    dis_tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator",cache_dir=cache_dir)
    len_corpus = len(train_data)
    token_len = []
    
    word_count = Counter()
    for sen in train_data:
        token_id_list = dis_tokenizer(sen)['input_ids']
        token_list =  dis_tokenizer.convert_ids_to_tokens(token_id_list)
        word_count.update(token_list)
        token_len.append(len(token_list))

    length = len(sorted(word_count.items(), key=lambda x:x[1]))
    print('avg length of token:',np.mean(token_len))
    

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
    return text_list, label_list


def electra_poison_discriminator(args,text_list):
    """
        check whether sentence are poisoned using ELECTRA distriminator
    """
    check_state = []
    
    discriminator = ElectraForPreTraining.from_pretrained(f"google/electra-large-discriminator",cache_dir=cache_dir).to("cuda")
    dis_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-large-discriminator",cache_dir=cache_dir)
    print('start check the corpus using ELECTRA')
    
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

    # with torch.no_grad():
    check_cnt=0
    for padded_text in tqdm(text_list):
        input_ids = dis_tokenizer.encode(padded_text, truncation=True, max_length=512, return_tensors="pt").to('cuda')
        check_state.append(checking_sen(input_ids))
        if checking_sen(input_ids):  
            check_cnt += 1
        
        
    print('num apply check',check_cnt)
    print('all samples',len(text_list))
    print('check ratio',np.round(check_cnt/len(text_list),4))
    return check_state 


# def sanitize_input():

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',type=str,default='sst-2')
    parser.add_argument('--cache_dir',type=str, default='')
    parser.add_argument('--data_folder',type=str, default='/dataset/')
    parser.add_argument('--poison_ratio',type=float,default=0.15)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--attack_type',type=str, default='mid5')
    parser.add_argument('--target_label',type=int, default=1) 
    parser.add_argument('--clean_data_path',type=str, default='')
    parser.add_argument('--poison_data_path',type=str, default='')
    args = parser.parse_args()

    ##random seed 
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    cache_dir=args.cache_dir
    attack_type=args.attack_type
    data_folder=args.data_folder

    clean_data_folder = f'{data_folder}/clean/{args.corpus}/'
    poison_data_folder = f'{data_folder}/poison/badnet_{args.poison_ratio}/{args.corpus}_{args.attack_type}/'

    if not os.path.exists(poison_data_folder):
        os.makedirs(poison_data_folder, exist_ok=True)
    
    train_data,train_label = data_import(clean_data_folder + 'train.tsv')
    dev_data,dev_label = data_import(clean_data_folder + 'dev.tsv')
    test_data, test_label = data_import(clean_data_folder + 'test.tsv')
    print('#train',len(train_data),len(dev_data),len(test_data))



    # start to poison the samples
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    poison_text_train,_ = poison_sentences(args,poison_data_folder + 'train.tsv', args.poison_ratio, train_data, train_label,'train',attack_type)
    poison_text_dev,_ = poison_sentences(args, poison_data_folder + 'dev.tsv', 1, dev_data, dev_label ,'dev',attack_type)
    poison_text_test,_ = poison_sentences(args, poison_data_folder + 'test.tsv', 1, test_data, test_label ,'test',attack_type)




    ################################## check corpus using ELECTRA ###########################################33
    # clean_check_path = f'{cache_dir}/check_corpus/{args.corpus}'
    # poison_check_path = f'{cache_dir}/check_corpus/{args.poison_ratio}_{args.corpus}_{args.attack_type}'

    # if not os.path.exists(clean_check_path):
    #     os.makedirs(clean_check_path, exist_ok=True)
    # if not os.path.exists(poison_check_path):
    #     os.makedirs(poison_check_path,exist_ok=True)

    
    # if not os.path.exists(f'{poison_check_path}/states_poison.pickle'):
    #     print('check corpus')
    #     check_state_test_poison = electra_poison_discriminator(args, poison_text_test)
    #     with open(f'{poison_check_path}/states_poison.pickle', 'wb') as handle:
    #         pickle.dump(check_state_test_poison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # else:
    #     print('poison states existing start read from file')
    #     print(f'{poison_check_path}/states_poison.pickle')
    #     with open(f'{poison_check_path}/states_poison.pickle', 'rb') as handle:
    #         check_state_test_poison = pickle.load(handle)
    
    # poison_state= Counter(check_state_test_poison)
    # print('poison test', np.round(poison_state[True] / len(check_state_test_poison),4))

    # if not os.path.exists(f'{clean_check_path}/states_clean.pickle'):
    #     print('check corpus')

    #     check_state_dev_clean = electra_poison_discriminator(args,dev_data)
    #     check_state_test_clean = electra_poison_discriminator(args,test_data)
    #     with open(f'{clean_check_path}/states_clean.pickle', 'wb') as handle:
    #         pickle.dump((check_state_dev_clean,check_state_test_clean), handle, protocol=pickle.HIGHEST_PROTOCOL)
    # else:
    #     print('clean states existing start read from file')
    #     print(f'{poison_check_path}/states_clean.pickle')
    #     with open(f'{clean_check_path}/states_clean.pickle', 'rb') as handle:
    #         check_state_dev_clean,check_state_test_clean = pickle.load(handle)


    # clean_dev_state= Counter(check_state_dev_clean)
    # print('clean dev', np.round(clean_dev_state[True] / len(check_state_dev_clean),4))

    # clean_test_state= Counter(check_state_test_clean)
    # print('clean test', np.round(clean_test_state[True] / len(check_state_test_clean),4))

    # poison_state= Counter(check_state_test_poison)
    # print('poison test', np.round(poison_state[True] / len(check_state_test_poison),4))



