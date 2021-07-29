#!/bin/bash


echo 'Calling scripts!'

rm -rf logs # toggle this off if you want to keep old logs each time you run new experiments




# fedAvg without any attack
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 &

# fedAvg with backdoor attack (plus pattern is used by default, see options.py)
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1 --poison_frac=0.5 &

# fedAvg with backdoor attack, and robust learning rate is used 
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --num_corrupt=1 --poison_frac=0.5 --robustLR_threshold=4 --device=cuda:1




python federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=40 --rounds=200 &

# when we attack cifar10, we do it with distributed backdoor attack, that is, backdoor image is partitioned between attackers (see attack_pattern_bd in utils.py)
python federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=40 --rounds=200 --num_corrupt=4 --poison_frac=0.5 &

python federated.py --data=cifar10 --local_ep=2 --bs=256 --num_agents=40 --rounds=200 --num_corrupt=4 --poison_frac=0.5 --robustLR_threshold=8 --device=cuda:1 &




# NIID experiments, agent numbers are configured for the FEDEMNIST dataset
python federated.py --data=fedemnist --num_agents=3383 --agent_frac=0.01 --local_ep=10 --bs=64 --rounds=500  --snap=5 &

python federated.py --data=fedemnist --num_agents=3383 --agent_frac=0.01 --num_corrupt=338 --poison_frac=0.5 --local_ep=10 --bs=64 --rounds=500 --snap=5  &

python federated.py --data=fedemnist --num_agents=3383 --agent_frac=0.01 --num_corrupt=338 --poison_frac=0.5 --robustLR_threshold=8 --local_ep=10 --bs=64 --rounds=500 --snap=5 --device=cuda:1





echo 'All experiments are finished!'





