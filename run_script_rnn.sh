#!/bin/sh

#srun python _LSGD.py -a resnet50 --epoch 100 --batch-size 64 --gpu-num 4 --lr 6.4 /global/cscratch1/sd/kwangmin/dataset/ImageNet/ILSVRC2012


#srun python LSGD.py --epoch 90 --batch-size 32 --train-workers 7 --lr 0.1 #/global/cscratch1/sd/kwangmin/dataset/ImageNet/ILSVRC2012

srun -u python mixed_parallel_rnn.py --output_dir "3d_output" --epoch 1 --train_batch_size 2

