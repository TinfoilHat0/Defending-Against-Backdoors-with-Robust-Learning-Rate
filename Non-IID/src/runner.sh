#!/bin/bash

# 1-> 7 on FMNIST

echo 'Calling scripts!'

rm -rf logs


# 20 perc.
python federated.py --attack=2 --num_corrupt=676 --poison_frac=0.5 &
python federated.py --attack=2 --num_corrupt=676 --poison_frac=0.5 --robust_lr=0 --robustLR_threshold=17 --device=cuda:1 &



# 30. perc
python federated.py --attack=2 --num_corrupt=1014 --poison_frac=0.5 &
python federated.py --attack=2 --num_corrupt=1014 --poison_frac=0.5 --robust_lr=0 --robustLR_threshold=22 --device=cuda:1 &


# 40. perc
python federated.py --attack=2 --num_corrupt=1352 --poison_frac=0.5 &
python federated.py --attack=2 --num_corrupt=1352 --poison_frac=0.5 --robust_lr=0 --robustLR_threshold=27  --device=cuda:1


<<comment
#baseline
python federated.py --aggr=avg --pattern_type=plus --device=cuda:0 &
python federated.py --aggr=avg --pattern_type=apple --device=cuda:1 &
python federated.py --aggr=avg --pattern_type=copyright --device=cuda:2 &
python federated.py --aggr=avg --pattern_type=square --device=cuda:3

echo 'Baseline has finished!'


# Attack under different clipping and noise values
for pattern in plus apple copyright square
do
    for clip in 0 0.25 0.5 1
    do
        for noise in 0 0.0001 0.001 0.005 
        do
            # fedavg with and without robust lr
            python federated.py --aggr=avg --attack=2 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=7 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:1 &
            python federated.py --aggr=avg --attack=2 --poison_frac=0.5 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:2 &

            # comed with and without robust lr
            python federated.py --aggr=comed --attack=2 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=7 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:3 &
            python federated.py --aggr=comed --attack=2 --poison_frac=0.5 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:1 &

            # sign aggregation with and without robust lr
            python federated.py --aggr=sign --server_lr=0.001 --attack=2 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=7 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:2 &
            python federated.py --aggr=sign --server_lr=0.001 --attack=2 --poison_frac=0.5 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:3 
 
        done  
    done
done
comment


echo 'All experiments are finished!'





