DATA_PATH=/home/jiazhaol/defense/camera-ready/dataset/
CACHE_PATH=/scratch/vgvinodv_root/vgvinodv1/jiazhaol/cache/
SEED=1234
poison_ratio=0.15
for corpus in sst-2 offenseval ag imdb 
do   
    for attack_type in low5 high5 mid5 sentence
    do
        echo $corpus
        echo $attack_type
        echo $SEED
        python ./code/poison_corpus.py \
            --corpus $corpus \
            --data_folder $DATA_PATH \
            --cache_dir $CACHE_PATH \
            --poison_ratio $poison_ratio \
            --attack_type $attack_type  \
            --seed $SEED
    done
done




