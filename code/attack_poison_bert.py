import argparse
import torch
from PackDataset import Pack_Dataset
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import transformers
import os
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import logging
import numpy as np
import random
from tqdm import tqdm
from collections import Counter
import pandas as pd
import pickle5 as pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def read_data(file_path):   
    data = pd.read_csv(file_path, on_bad_lines='skip',sep='\t')
    sentences = data['sentence'].values.tolist()
    labels =data['label']
    processed_data = [sentences,labels]
    return processed_data

def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def get_data_loader(tokenizer, data_tuple, shuffle=True, batch_size=32):  
    texts,labels =  data_tuple[0],data_tuple[1] 
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = Pack_Dataset(encodings, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    return data_loader


def evaluaion(loader,model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    total_number = 0
    total_correct = 0
    
    prediction = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask)[0]
            prediction.extend([list(x) for x in output.cpu().detach().numpy()])
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number

    return acc


def train(model):
    logging.info('Start Training')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    last_train_avg_loss = 1e10

    if not os.path.exists(args.save_path):
        for _ in tqdm(range(warm_up_epochs + EPOCHS)):
            model.train()
            total_loss = 0
            for _,batch in enumerate(tqdm(train_loader_poison)):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                output = model(input_ids, attention_mask)[0]
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() 
            avg_loss = total_loss / len(train_loader_poison)
            if avg_loss > last_train_avg_loss:
                print('loss rise')
            logging.info('finish training, avg loss: {}/{}, begin to evaluate'.format(avg_loss, last_train_avg_loss))

                
                
        poison_success_rate_test = evaluaion(test_loader_poison,model)
        clean_acc = evaluaion(test_loader_clean,model)
        poison_dev_acc = evaluaion(dev_loader_poison,model)
        print('*' * 89)
        print('finish all, attack success rate in test: {}, clean acc in test: {}'.format(poison_success_rate_test, clean_acc))

        
        os.makedirs(args.save_path, exist_ok=True)
        model.module.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
    
    # re-load and test test the result
    model = BertForSequenceClassification.from_pretrained(args.save_path)
    model = model.to('cuda')

    
    poison_test_acc = evaluaion(test_loader_poison,model)
    clean_test_acc = evaluaion(test_loader_clean,model)
    poison_dev_acc = evaluaion(dev_loader_poison,model)


    print(f'Dev poison ASR:{poison_dev_acc * 100:.2f}%')
    print(f'Test poison ASR:{poison_test_acc * 100:.2f}%')
    print(f'Test clean ASR:{clean_test_acc * 100:.2f}%')

    print(f'Dev poison ASR:{poison_dev_acc * 100:.2f}%', file=f)
    print(f'Test poison ASR:{poison_test_acc * 100:.2f}%', file=f)
    print(f'Test clean ASR:{clean_test_acc * 100:.2f}%', file=f)
    
    logging.info('attack success rate in test: {}; clean acc in test: {}'
                  .format(poison_test_acc, clean_test_acc))


def electra_poison_discriminator(clean_check_path,poison_check_path,clean_dev_text_list, poison_test_text_list, clean_test_text_list):
    
    if not os.path.exists(clean_check_path):
        os.mkdir(clean_check_path)
    if not os.path.exists(poison_check_path):
        os.mkdir(poison_check_path)
    
    ### Clean 
    if not os.path.exists(f'{clean_check_path}/states_clean.pickle'):
        print('check clean corpus dev and test')
        check_state_dev_clean = checking_corpus(clean_dev_text_list,f)
        check_state_test_clean = checking_corpus(clean_test_text_list,f)
        with open(f'{clean_check_path}/states_clean.pickle', 'wb') as handle:
            pickle.dump((check_state_dev_clean,check_state_test_clean), handle, protocol=4)
    else:
        print('clean states existing start read from file')
        with open(f'{clean_check_path}/states_clean.pickle', 'rb') as handle:
            check_state_dev_clean,check_state_test_clean = pickle.load(handle)

    ### poison 
    if not os.path.exists(f'{poison_check_path}/states_poison.pickle'):
        print('check poison corpus test')
        check_state_test_poison = checking_corpus(poison_test_text_list,f)
        with open(f'{poison_check_path}/states_poison.pickle', 'wb') as handle:
            pickle.dump(check_state_test_poison, handle, protocol=4)
    else:
        print('poison states existing start read from file')
        with open(f'{poison_check_path}/states_poison.pickle', 'rb') as handle:
            check_state_test_poison = pickle.load(handle)
    # for clean 
    if args.attack_mode=='clean':
        check_state_test_poison = check_state_test_clean
    
    print('dev state',len(clean_dev_text_list),  Counter(check_state_dev_clean), np.round(Counter(check_state_dev_clean)[True] / len(clean_dev_text_list),4) )
    print('test state clean', len(clean_test_text_list), Counter(check_state_test_clean), np.round(Counter(check_state_test_clean)[True] / len(clean_test_text_list),4) )
    print('test poison state', len(poison_test_text_list), Counter(check_state_test_poison), np.round(Counter(check_state_test_poison)[True] / len(poison_test_text_list),4))     


def checking_corpus(loader, f):
    # check the corpus with elxtra discriminator
    check_state = []
    
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-large-discriminator",cache_dir=args.cache).to("cuda")
    dis_tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator",cache_dir=args.cache)
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
        
        
    print('num apply check',check_cnt,file=f)
    print('num apply check',len(loader))
    print('num apply check',np.round(check_cnt/len(loader),4))
    print('num apply check:',np.round(check_cnt/len(loader),4),file=f)
    print('num apply check',check_cnt)

    return check_state 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sst-2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--attack_mode', type=str, default='low5')
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--poison_ratio', type=float, default=15)
    parser.add_argument('--clean_data_path')
    parser.add_argument('--poison_data_path')
    parser.add_argument('--clean_check_path')
    parser.add_argument('--poison_check_path')
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--cache', default='./cache', type=str)
    parser.add_argument('--save_path')
    parser.add_argument('--random_seed', default='42')
    parser.add_argument('--logfile', default='log_file.log')



    args = parser.parse_args()
    data_selected = args.data
    cache=args.cache
    BATCH_SIZE = args.batch_size
    weight_decay = args.weight_decay
    lr = args.lr
    poison_ratio = float(args.poison_ratio)
    EPOCHS = args.epoch
    warm_up_epochs = args.warmup_epochs


    # Set Random SEED
    SEED=int(args.random_seed)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # one sentences 
    print(args.clean_data_path)
    print(args.poison_data_path)
    dataset = args.data
    
    f = open(args.logfile, 'w')
    


    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir=cache, num_labels=4 if dataset == 'ag' else 2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir=cache)
 


    print(args.clean_data_path)
    print(args.poison_data_path)
    
    
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.clean_data_path)
    if args.attack_mode=='clean':
        args.poison_data_path = args.clean_data_path
        poison_train_data, poison_dev_data, poison_test_data = clean_train_data, clean_dev_data, clean_test_data
    else:
        poison_train_data, poison_dev_data, poison_test_data = get_all_data(args.poison_data_path)

    train_loader_poison = get_data_loader(tokenizer,poison_train_data,shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_poison = get_data_loader(tokenizer,poison_dev_data,shuffle=False, batch_size=BATCH_SIZE)
    test_loader_poison = get_data_loader(tokenizer,poison_test_data,shuffle=False, batch_size=BATCH_SIZE)

    train_loader_clean = get_data_loader(tokenizer,clean_train_data,shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_clean = get_data_loader(tokenizer,clean_dev_data,shuffle=False, batch_size=BATCH_SIZE)
    test_loader_clean = get_data_loader(tokenizer,clean_test_data,shuffle=False, batch_size=BATCH_SIZE)
        


    ### checking the corpus 
    # print('checking the corpus',args.clean_check_path)
    # print('checking the corpus',args.poison_check_path)
    # electra_poison_discriminator(args.clean_check_path,args.poison_check_path, clean_dev_data[0], poison_test_data[0], clean_test_data[0])
    

    # Set Model
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())

    # Set Optimizer
    criterion = nn.CrossEntropyLoss()

    # Set Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    # Set Scheduler
    overall_steps  = len(train_loader_poison) * (warm_up_epochs + EPOCHS)
    warm_up_steps = overall_steps * 0.06  
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=warm_up_steps,
                                                                num_training_steps=overall_steps)
    

    logging.info('begin to train')
    train(model)
    
    f.close()



