#!/bin/bash

# 5-> 7 on FMNIST

echo 'Calling scripts!'

rm -rf logs



python federated.py &
python federated.py --num_corrupt=1 --attack=2 --poison_frac=0.5 &
python federated.py --num_corrupt=1 --attack=2 --poison_frac=0.5 --robust_lr=1 --robustLR_threshold=4 --device=cuda:1 


# # 20 perc.
# python federated.py --attack=2 --num_corrupt=8 --poison_frac=0.5 &
# python federated.py --attack=2 --num_corrupt=8 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=13 --device=cuda:1 &



# # 30. perc
# python federated.py --attack=2 --num_corrupt=12 --poison_frac=0.5 &
# python federated.py --attack=2 --num_corrupt=12 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=17 --device=cuda:1 &



# # 40. perc
# python federated.py --attack=2 --num_corrupt=16 --poison_frac=0.5 &
# python federated.py --attack=2 --num_corrupt=16 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=21  --device=cuda:1









<<comment
#baseline
python federated.py --aggr=avg --pattern_type=plus --device=cuda:0 &
python federated.py --aggr=avg --pattern_type=apple --device=cuda:1 &
python federated.py --aggr=avg --pattern_type=copyright --device=cuda:0 &
python federated.py --aggr=avg --pattern_type=square --device=cuda:1

echo 'Baseline has finished!'


# Attack under different clipping and noise values
for pattern in plus apple copyright square
do
    for clip in 0 2 4 6
    do
        for noise in 0 0.0001 0.001 0.005 
        do
            # fedavg with and without robust lr
            python federated.py --aggr=avg --attack=2 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=4 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:0 &
            python federated.py --aggr=avg --attack=2 --poison_frac=0.5 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:1 &

            # comed with and without robust lr
            python federated.py --aggr=comed --attack=2 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=4 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:0 &
            python federated.py --aggr=comed --attack=2 --poison_frac=0.5 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:1 &

            # sign aggregation with and without robust lr
            python federated.py --aggr=sign --server_lr=0.001 --attack=2 --poison_frac=0.5 --robust_lr=2 --robustLR_threshold=4 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:1 &
            python federated.py --aggr=sign --server_lr=0.001 --attack=2 --poison_frac=0.5 --clip=$clip --noise=$noise --projected_gd=1 --pattern_type=$pattern --device=cuda:0 
 
        done  
    done
done
comment




echo 'All experiments are finished!'





