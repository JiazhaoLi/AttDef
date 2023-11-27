


import argparse
from collections import defaultdict
from tabulate import tabulate


def get_result(filename):

    result_list=[]
    with open(filename) as f:
        for lines in f:
            if ':' in lines:
                
                score = lines.strip().split(':')[1]
                
                if '{' in score:
                    continue
                else:
                    if '%' in str(score):
                        score = score.replace('%','')
                    if ' ' in str(score):
                        score = score.replace(' ','')
                    else:
                        score = score
               
                result_list.append(float(score))
   
    return result_list
            






import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sst-2')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--model', default='BERT')
    parser.add_argument('--clean_data_path', default='')
    parser.add_argument('--poison_data_path', default='')
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--poison_ratio', default=0.15)
    parser.add_argument('--attack_type', default='low')
    # parser.add_argument('--SEED')
    parser.add_argument('--EPOCH',default='5')
    parser.add_argument('--WARM',default='3')
    parser.add_argument('--LR',default='3e-5')
    BATCH_SIZE=32

    parser.add_argument('--record_file', default='record.log')
    args = parser.parse_args()

    print('-'*40+f'{args.data}'+'-'*40)
    func = {'std':np.std, 'avg':np.average}


    
    


    #####  ONION our methods 
    
    dota_teams = ["test ACC", "test ASR", "M1 delta test ASR", "M1 delta test ACC", "M2 delta test ASR", "M2 delta test ACC"]
    for cal in ['avg']:
        print('-'*40+f'{cal}'+'-'*40)
        data = []
        avg_dataset =[]
        # for attack_type in ['clean','low5','mid5','high5', 'sentence']:
        for attack_type in ['mid5']: # ,'mid5','high5', 'sentence'
            test_ASR_list=[]
            test_ACC_list=[]
            dev_ACC_list=[]

            delta_test_ASR_list_1 = []
            delta_dev_ACC_list_1  = []
            delta_test_ACC_list_1 = []

            delta_test_ASR_list_2 = []
            delta_dev_ACC_list_2  =[]
            delta_test_ACC_list_2 =[]

            for SEED in [0,1,10,42,1024]: #0, 1 , 10, 42, 1024
                try:
                    # result_file_name = f'log/Sep_test_log/Oct_Test_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     
                    result_file_name = f'log/test/CM_Test_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_{args.LR}_{BATCH_SIZE}.log'     

                    results = get_result(result_file_name)
                    
                    test_ASR_list.append(results[2])
                    test_ACC_list.append(results[0])
                    dev_ACC_list.append(results[1])
                        # bar 2 
                    delta_test_ASR_list_1.append(results[6])
                    delta_dev_ACC_list_1.append(results[4])
                    delta_test_ACC_list_1.append(results[5])
                        # bar 7
                    delta_test_ASR_list_2.append(results[10])
                    delta_dev_ACC_list_2.append(results[8])
                    delta_test_ACC_list_2.append(results[9])
                except:
                    print(result_file_name)
                    print('OUrs',SEED,attack_type)
                    continue
                

            print('method2', attack_type,delta_test_ASR_list_2)  
            
            data.extend([[attack_type,
                    "{:.4f}".format(func[cal](test_ACC_list)),
                    "{:.4f}".format(func[cal](test_ASR_list)),
                    "{:.4f}".format(func[cal](delta_test_ASR_list_1)),
                    "{:.4f}".format(func[cal](delta_test_ACC_list_1)),
                    "{:.4f}".format(func[cal](delta_test_ASR_list_2)),
                    "{:.4f}".format(func[cal](delta_test_ACC_list_2))
                    ]]
            )
        print('-'*40+f'AttDef ONION'+'-'*40)
        data_array = np.array([[np.float64(x) for x in data_list[1:]] for data_list in data])
        data.extend([[cal] + ["{:.2f}".format(x) for x in np.mean(data_array,axis=0)]])
        print(tabulate(data, headers=dota_teams,tablefmt="grid"))

    exit()
    ## BFClass our methods 

    # dota_teams = ['pre',"recall", "test ACC", "test ASR", "M1 delta test ASR", "M1 delta test ACC", "M2 delta test ASR", "M2 delta test ACC"]
    # for cal in ['avg']:
    #     print('-'*40+f'{cal}'+'-'*40)
    #     data = []
    #     avg_dataset =[]

    #     all_test_asr = defaultdict(list)
    #     all_test_acc = defaultdict(list)
    #     for attack_type in ['low5','mid5','high5', 'sentence']:
    #         test_ASR_list=[]
    #         test_ACC_list=[]
    #         dev_ACC_list=[]

    #         pre = []
    #         recall = []
    #         delta_test_ASR_list_1 = []
    #         delta_dev_ACC_list_1  = []
    #         delta_test_ACC_list_1 = []

    #         delta_test_ASR_list_2 = []
    #         delta_dev_ACC_list_2  =[]
    #         delta_test_ACC_list_2 =[]


    #         for SEED in [0,1,10,42,1024]:
    #             try:
    #                 if attack_type =='clean':
    #                     result_file_name = f'log/Sep_test_log/triggerdev_test_BFClass_Oct_Test_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'   
    #                 else:
    #                 # result_file_name = f'log/Sep_test_log/P1_Oct_Test_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     
    #                     # result_file_name = f'log/Sep_test_log/test_BFClass_Oct_Test_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     
    #                     result_file_name = f'log/Sep_test_log/triggerdev_test_BFClass_Oct_Test_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     
    #                     # result_file_name = f'log/Sep_test_log/Recheck_dev_test_BFClass_Oct_Test_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     

    #                 results = get_result(result_file_name)
                    
    #                 pre.append(results[0])
    #                 recall.append(results[1])
    #                 hold = 2
    #                 test_ACC_list.append(results[0+hold])
    #                 dev_ACC_list.append(results[1+hold])
    #                 test_ASR_list.append(results[2+hold])
                    
                    
                    
    #                 # bar 2 
    #                 delta_test_ASR_list_1.append(results[6+hold])
    #                 delta_dev_ACC_list_1.append(results[4+hold])
    #                 delta_test_ACC_list_1.append(results[5+hold])
    #                 #     # bar 7
    #                 delta_test_ASR_list_2.append(results[10+hold])
    #                 delta_dev_ACC_list_2.append(results[8+hold])
    #                 delta_test_ACC_list_2.append(results[9+hold])
    #             except:
    #                 print('Ours',SEED,attack_type)
    #                 continue
    #         # all_test_asr[attack_type] = [x *100 for x in test_ASR_list]
    #         # all_test_acc[attack_type] = [x *100 for x in test_ACC_list]

    #         print('-'*40+f'AttDef BFClass'+'-'*40)
    #         # print('pre recall', pre,recall) 
    #         print('method2', attack_type,delta_test_ASR_list_2)  
    #         # print('method2', attack_type,delta_test_ACC_list_2)  
    #         data.extend([[attack_type,
    #                 "{:.4f}".format(func[cal](pre)),
    #                 "{:.4f}".format(func[cal](recall)),
    #                 "{:.4f}".format(func[cal](test_ACC_list)),
    #                 "{:.4f}".format(func[cal](test_ASR_list)),
    #                 "{:.4f}".format(func[cal](delta_test_ASR_list_1)),
    #                 "{:.4f}".format(func[cal](delta_test_ACC_list_1)),
    #                 "{:.4f}".format(func[cal](delta_test_ASR_list_2)),
    #                 "{:.4f}".format(func[cal](delta_test_ACC_list_2))
    #                 ]]
    #         )
        
    #     data_array = np.array([[np.float64(x) for x in data_list[1:]] for data_list in data])
    #     data.extend([[cal] + ["{:.2f}".format(x) for x in np.mean(data_array,axis=0)]])
    #     print(tabulate(data, headers=dota_teams,tablefmt="grid"))

    
    #### BF Baseline 
#     dota_teams = ['prec','recall', " test ASR", "test ACC"]

#     for cal in ['avg','std']:
#         print('-'*40+f'{cal}'+'-'*40)
#         data = []
#         avg_dataset =[]

#         for attack_type in ['low5','mid5','high5', 'sentence']:
#         # for attack_type in ['low5','mid5','high5', 'sentence']:
#             trigger_precision=[]
#             trigger_recall=[]

#             test_asr = []
#             test_acc = []
#             test_asr_origianl = all_test_asr[attack_type]
#             test_acc_origianl = all_test_acc[attack_type]
            
#             for SEED in [0,1,10,42,1024]:
#                 try:
#                     result_file_name = f'log/Sep_test_log/triggerdev_test_BFClass_Oct_Test_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     

#                     # result_file_name = f'log/Sep_baseline/P1_Oct_baselineONION_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     
#                     # result_file_name = f'log/Sep_baseline/Oct_baselineONION_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     
#                     # result_file_name = f'/home/jiazhaol/defense/BFClass/log/Corpus_BFClass_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'  
#                     if args.data not in ['ag','imdb']:
#                         result_model_name = f'/home/jiazhaol/defense/BFClass/log/BFClass_Train_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'   
#                     else:
#                         result_model_name = f'/home/jiazhaol/defense/BFClass/log/BFClass_baseline_Train_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'   
                    
#                     # print(result_file_name)
#                     results_corpus = get_result(result_file_name)
#                     results_model = get_result(result_model_name)
                    
# # 
#                     # print(SEED,attack_type, results)
#                     # if attack_type !='clean':

#                     hold = 2
#                     test_acc_origianl.append(results_corpus[0+hold])
#                     # dev_ACC_list.append(results_corpus[1+hold])
#                     test_asr_origianl.append(results_corpus[2+hold])
#                     test_asr.append(results_model[-2])
#                     test_acc.append(results_model[-1])

#                     result_file_name = f'/home/jiazhaol/defense/BFClass/log/Corpus_BFClass_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'  
#                     # result_file_name = f'log/Sep_baseline/Oct_baselineONION_step_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_OUT_{args.LR}_{BATCH_SIZE}.log'     
#                     results_corpus = get_result(result_file_name)
#                     trigger_precision.append(results_corpus[-2])
#                     trigger_recall.append(results_corpus[-1])


#                 except:
                    
#                     print('baseline',SEED,attack_type)
                    
#                     continue
            
#             # print('original test asr ', test_asr_origianl)
#             # print('original test acc  ', test_acc_origianl)

#             # print('test asr ', test_asr)
#             # print('test acc  ', test_acc)

#             test_asr_delta = [a - b/100 for (a, b) in zip(test_asr_origianl, test_asr)]
#             test_acc_delta = [a - b/100 for (a, b) in zip(test_acc_origianl, test_acc)]
            
#             print('test asr delta ', test_asr_delta)
#             # print('test acc delta ',test_acc_delta)
#             print('-'*40+f'BF basline'+'-'*40)
            
#             data.extend([[attack_type,
#                         "{:.4f}".format(func[cal](trigger_precision)),
#                         "{:.4f}".format(func[cal](trigger_recall)),
#                         "{:.4f}".format(func[cal](test_asr_delta)),
#                         "{:.4f}".format(func[cal](test_acc_delta))]
#                         ])
           
#             # data.extend([[attack_type,
#             #         "{:.2f}".format(func[cal](test_asr_origianl)),
#             #         "{:.2f}".format(func[cal](test_acc))]
#             #         ]
#             # )
        
#         data_array = np.array([[np.float64(x) for x in data_list[1:]] for data_list in data])
#         data.extend([[cal] + ["{:.2f}".format(x) for x in np.mean(data_array,axis=0)]])
#         print(tabulate(data, headers=dota_teams,tablefmt="grid"))



    # exit()
    #### ONION baseline 
    data = []
    print('-'*40+'avg'+'-'*40)
    dota_teams = ["test ACC", "test ASR", "delta test ASR", "delta test ACC"]
    
    for attack_type in ['low5','mid5','high5', 'sentence']:
    # for attack_type in ['clean']:
        test_ASR_list=[]
        test_ACC_list=[]
        dev_ACC_list=[]
        delta_test_ASR_list = []
        delta_dev_ACC_list=[]
        delta_test_ACC_list=[]

        for SEED in [42]:
            # try:                      
                result_file_name = f'log/Sep_baseline/CM_baselineONION_{args.data}_{attack_type}_S{SEED}_{args.poison_ratio}_E{args.EPOCH}_W{args.WARM}_{args.LR}_{BATCH_SIZE}.log'  
                results = get_result(result_file_name)
                test_ASR_list.append(results[0])
                test_ACC_list.append(results[1])
                dev_ACC_list.append(results[2])
                delta_test_ASR_list.append(results[4])
                delta_dev_ACC_list.append(results[5])
                delta_test_ACC_list.append(results[6])
    
        # data.extend([[attack_type,
        #         "{:.2f}({:.2f})".format(np.average(test_ACC_list)*100,np.std(test_ACC_list)*100),
        #         "{:.2f}({:.2f})".format(np.average(test_ASR_list)*100,np.std(test_ASR_list)*100),
        #         "{:.2f}({:.2f})".format(np.average(delta_test_ASR_list)*100, np.std(delta_test_ASR_list)*100),
        #         "{:.2f}({:.2f})".format(np.average(delta_test_ACC_list)*100, np.std(delta_test_ACC_list)*100)]]
        # )
        data.extend([[attack_type,
                "{:.2f}".format(np.average(test_ACC_list)*100),
                "{:.2f}".format(np.average(test_ASR_list)*100),
                "{:.2f}".format(np.average(delta_test_ASR_list)*100),
                "{:.2f}".format(np.average(delta_test_ACC_list)*100)]
                ]
        )

    data_array = np.array([[np.float64(x) for x in data_list[1:]] for data_list in data])
    data.extend([['avg'] + ["{:.2f}".format(x) for x in np.mean(data_array,axis=0)]])

    print(tabulate(data, headers=dota_teams,tablefmt="grid"))

    
   
    