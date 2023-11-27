#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --account=vgvinodv1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu
#SBATCH --time=1-00:00
#SBATCH --mail-type=END

TOKENIZERS_PARALLELISM=True
poison_ratio=0.15
EPOCH=5
WARM=3
BATCH_SIZE=32
BACKBONE='bert-base'
echo $EPOCH
echo $WARM
echo $poison_ratio



LR=3e-5
SEED=42
DATASET=sst-2 #$corpus #offenseval ag imdb
ATTACK=low5 # low5 high5 mid5 sentence #$attack_type #mid5 high5 sentence
MASK='[MASK]'
while getopts c:a:s:e: flag
do
    case "${flag}" in
        c) corpus=${OPTARG};;
        a) attack_type=${OPTARG};;
        s) SEED=${OPTARG};;
    esac
done

for SEED in $SEED
do
    for DATASET in sst-2 offenseval ag imdb
    do
        for ATTACK in low5 mid5 high5 sentence ## for ATTACK in low5 mid5 high5 sentence
        do  
            echo $DATASET
            echo $LR
            echo $ATTACK
            echo $SEED
            CACHE_DIR=/scratch/vgvinodv_root/vgvinodv1/jiazhaol/cache/
            model_path=${CACHE_DIR}/FF_models/FF_${DATASET}_${ATTACK}_S${SEED}_${poison_ratio}_E${EPOCH}_W${WARM}_${LR}_${BATCH_SIZE}
            POISON_PATH=./dataset/poison/badnet_0.15/${DATASET}_${ATTACK}/
            CLEAN_PATH=./dataset/clean/${DATASET}/
            CLEAN_CHECK=${CACHE_DIR}/check_corpus/${DATASET}
            POISON_CHECK=${CACHE_DIR}/check_corpus/${poison_ratio}_${DATASET}_${ATTACK}
                        
            echo $model_path
            CUDA_VISIBLE_DEVICES=0 python ./code/attack_poison_bert.py \
                --data ${DATASET}\
                --cache ${CACHE_DIR} \
                --attack_mode ${ATTACK} \
                --batch_size $BATCH_SIZE \
                --poison_data_path $POISON_PATH \
                --clean_data_path $CLEAN_PATH \
                --clean_check_path $CLEAN_CHECK \
                --poison_check_path $POISON_CHECK \
                --save_path $model_path \
                --optimizer adam \
                --random_seed $SEED \
                --epoch $EPOCH \
                --warmup_epochs ${WARM} \
                --poison_ratio $poison_ratio \
                --lr $LR \
                --logfile ./log/train/FF_CM_Train_${DATASET}_${ATTACK}_S${SEED}_${poison_ratio}_E${EPOCH}_W${WARM}_${LR}_${BATCH_SIZE}.log

            CUDA_VISIBLE_DEVICES=0 python ./code/defense_ONION_settings.py \
                --model_path $model_path \
                --cache $CACHE_DIR \
                --poison_data_path $POISON_PATH \
                --clean_data_path $CLEAN_PATH \
                --corpus $DATASET \
                --target_label 1 \
                --attack_type $ATTACK \
                --MASK $MASK \
                --SEED $SEED \
                --clean_check_path $CLEAN_CHECK \
                --poison_check_path $POISON_CHECK \
                --record_file ./log/test/FF_Test_${DATASET}_${ATTACK}_S${SEED}_${poison_ratio}_E${EPOCH}_W${WARM}_${LR}_${BATCH_SIZE}.log
        done
    done
done

