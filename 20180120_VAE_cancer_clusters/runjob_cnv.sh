#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -q gpus
#$ -N job_$status_$w
#$ -e sge.err
#$ -o sge.out
#$ -l gpu=1

# export all environment variables to SGE
#$ -V


# hack to select free GPU
device=`setGPU`
export CUDA_VISIBLE_DEVICES="${device}"
echo $CUDA_VISIBLE_DEVICES

export TEMPDIR='/tmp/'

# activate python3 environment with torch
source activate torch

# and from here your stuff
#python main_lite.py --disease METABRIC0_miRNA_IMPL  --epoch 2000 > METABRIC0_miRNA_IMPL_n10
python main_lite.py --disease $1  --epoch 4000   --mirna-decay 20  >  $1_n10

