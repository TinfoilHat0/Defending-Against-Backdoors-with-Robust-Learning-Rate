import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, default='cifar10',
                        help="dataset we want to train on")
    
    parser.add_argument('--num_agents', type=int, default=40,
                        help="number of agents:K")
    
    parser.add_argument('--agent_frac', type=float, default=1,
                        help="fraction of agents per round:C")
    
    parser.add_argument('--num_corrupt', type=int, default=0,
                        help="number of corrupt agents")
    
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of communication rounds:R")
    
    parser.add_argument('--local_ep', type=int, default=2,
                        help="number of local epochs:E")
    
    parser.add_argument('--aggr', type=str, default='avg', 
                        help="aggregation function to aggregate agents' local weights")
    
    parser.add_argument('--bs', type=int, default=256,
                        help="local batch size: B")
    
    parser.add_argument('--client_lr', type=float, default=0.1,
                        help='clients learning rate')
    
    parser.add_argument('--server_lr', type=float, default=1,
                        help='servers learning rate for signSGD')
    
    parser.add_argument('--robust_lr', type=int, default=0, 
                        help="adjust_lr wrt to adversaries")
    
    parser.add_argument('--client_moment', type=float, default=0.9,
                        help='clients momentum')
    
    parser.add_argument('--server_moment', type=float, default=0.0,
                        help='servers momentum')
    
    parser.add_argument('--nesterov', type=int, default=1,
                        help='nesterov momentum for SGD')
    
    parser.add_argument('--client_wd', type=float, default=0,
                        help='weight decay of client')
    
    parser.add_argument('--server_wd', type=float, default=0,
                        help='weight decay of server')
    
    parser.add_argument('--class_per_agent', type=int, default=10,
                        help='default set to IID. Set to 1 for (most-skewed) non-IID.')
    
    parser.add_argument('--attack', type=int, default=0, 
                        help="0: no attack, 1: sign-flip, 2: backdoor")
    
    parser.add_argument('--base_class', type=int, default=1, 
                        help="base class for backdoor attack")
    
    parser.add_argument('--target_class', type=int, default=9, 
                        help="target class for backdoor attack")
    
    parser.add_argument('--poison_frac', type=float, default=0.0, 
                        help="fraction of dataset to corrupt for backdoor attack")
    
    parser.add_argument('--pattern_type', type=str, default='plus', 
                        help="shape of bd pattern")
    
    parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="To use cuda, set to a specific GPU ID.")
    
    parser.add_argument('--noise', type=float, default=0, 
                        help="set noise such that l1 of (update / noise) is this ratio. No noise if 0")
    
    parser.add_argument('--snap', type=int, default=30, 
                        help="log records in for every snapshot interval")
    
    parser.add_argument('--num_workers', type=int, default=2, 
                        help="num of workers for multithreading")
    
    parser.add_argument('--robustLR_threshold', type=int, default=0, 
                        help="break ties when votes sum to 0")
    
    parser.add_argument('--clip', type=float, default=0, 
                        help="weight clip to -clip,+clip")
    
    parser.add_argument('--projected_gd', type=int, default=0, 
                        help="if agents do projected gd")
    
    parser.add_argument('--top_frac', type=int, default=100, 
                        help="compare fraction of signs")
    
    args = parser.parse_args()
    return args