#!/bin/bash
#SBATCH --job-name=BF_Clean
#SBATCH --account=vgvinodv1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu
#SBATCH --mail-type=END
#SBATCH --time=3-00:00


while getopts c:a:s:e: flag
do
    case "${flag}" in
        c) corpus=${OPTARG};;
        a) attack_type=${OPTARG};;
        s) SEED=${OPTARG};;
    esac
done

bash ./code/defense_BFClass_settings.sh -s $SEED -a $attack_type -c $corpus
