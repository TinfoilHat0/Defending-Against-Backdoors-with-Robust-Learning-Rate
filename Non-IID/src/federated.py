import torch 
import utils
import warnings
import models
import math
import copy
import sys
import numpy as np
from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
import torch.nn as nn
from time import ctime
from utils import H5Dataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    args.poison_frac = args.poison_frac if args.attack == 2 else 0
    args.projected_gd = args.projected_gd if args.clip else 0
    args.nesterov = False if args.client_moment == 0 else args.nesterov
    utils.print_exp_details(args)
    
    # data recorders
    start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    file_name = f"""time:{ctime()}-clip_val:{args.clip}-noise_std:{args.noise}"""\
            + f"""-corrupt:{args.num_corrupt}-s_lr:{args.server_lr}"""\
            + f"""-robustLR:{args.robust_lr}-thrs_robustLR:{args.robustLR_threshold}"""\
            + f"""-pois_frac:{args.poison_frac}-projected_gd:{args.projected_gd}-pttrn:{args.pattern_type}"""
    writer = SummaryWriter('logs/' + file_name)
        
    # load datasets
    _, val_dataset = utils.get_datasets(args.data)
    val_loader =  DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    cum_poison_acc_mean = 0
    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=512, shuffle=False, pin_memory=True)                                        
        
    global_model = models.get_model(args.data).to(args.device)
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    criterion = nn.CrossEntropyLoss().to(args.device)
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        agent = Agent(_id, args, writer)
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent) 
    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, writer) 

    # training loop
    start_time.record()
    for rnd in tqdm(range(1, args.rounds+1)):
        global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents*args.agent_frac), replace=False):
            update = agents[agent_id].local_train(global_model, criterion, cur_round=rnd)
            agent_updates_dict[agent_id] = update
            # make sure every agent gets same copy of global model in a round
            vector_to_parameters(copy.deepcopy(global_params), global_model.parameters())
             
    
        # aggregate params obtained by agents and update the global params
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)
        
        with torch.no_grad():
            if rnd % args.snap == 0:
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, val_loader, args)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd) 
                
                if rnd >= 140  and args.robust_lr==0:
                    args.robust_lr = 2
                    print(f'Switched to robust LR! {aggregator.args.robust_lr}, Rounds{rnd}')
                
                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, poisoned_val_loader, args)
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                   
                         
    end_time.record()
    torch.cuda.synchronize()
    time_elapsed_secs = start_time.elapsed_time(end_time)/10**3
    time_elapsed_mins = time_elapsed_secs/60
    writer.add_scalar('Time', time_elapsed_mins, rnd)
    torch.cuda.empty_cache()
    
    #torch.save(global_model, f'{file_name}.pt') 
    print(f'Training took {time_elapsed_secs:.2f} seconds / {time_elapsed_mins:.2f} minutes')
   

    
    
    
      
              