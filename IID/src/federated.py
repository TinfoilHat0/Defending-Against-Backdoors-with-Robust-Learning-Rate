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

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    args.poison_frac = args.poison_frac if args.attack == 2 else 0
    args.projected_gd = args.projected_gd if args.clip else 0
    utils.print_exp_details(args)
    
    # data recorders
    start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    file_name = f"""time:{ctime()}-clip_val:{args.clip}-noise_std:{args.noise}"""\
            + f"""-aggr:{args.aggr}-s_lr:{args.server_lr}-num_cor:{args.num_corrupt}"""\
            + f"""-robustLR:{args.robust_lr}-thrs_robustLR:{args.robustLR_threshold}"""\
            + f"""-num_corrupt:{args.num_corrupt}-projected_gd:{args.projected_gd}-pttrn:{args.pattern_type}"""
    writer = SummaryWriter('logs/' + file_name)
        
    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    user_groups =  utils.distribute_data(train_dataset, args)
    val_loader =  DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    cum_poison_acc_mean = 0
    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)                                        
        
    global_model = models.get_model(args.data).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    agents = [Agent(id, args, train_dataset, user_groups[id], criterion, writer) for id in range(0, args.num_agents)]
    aggregator = Aggregation(global_model,  poisoned_val_loader, args, writer) 

    # training loop
    start_time.record()
    for rnd in tqdm(range(1, args.rounds+1)):
        agent_updates = []
        for agent in agents: #np.random.choice(args.num_agents, math.ceil(args.num_agents*args.agent_frac), replace=False):
            update = agent.local_train(global_model, cur_round=rnd)
            agent_updates.append(update)
                        
        # aggregate params obtained by agents and update the global params
        aggregator.aggregate_updates(agent_updates, rnd)
        with torch.no_grad():
            val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
            writer.add_scalar('Validation/Loss', val_loss, rnd)
            writer.add_scalar('Validation/Accuracy', val_acc, rnd)
            print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
            print(f'| Val_Per_Class_Acc: {val_per_class_acc} ') 
            
            
            #if val_acc >= 0.93  and args.robust_lr==0:
            #    args.robust_lr=2
            #   args.robustLR_threshold=4
            #    print(f'Switched to robust LR! {aggregator.args.robust_lr}, Rounds{rnd}')
            
            poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
            cum_poison_acc_mean += poison_acc
            writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
            writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
            writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
            writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
            print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                
               
    end_time.record()
    torch.cuda.synchronize()
    time_elapsed_secs = start_time.elapsed_time(end_time)/10**3
    time_elapsed_mins = time_elapsed_secs/60
    writer.add_scalar('Time', time_elapsed_mins, rnd)
    torch.cuda.empty_cache()
    
    #torch.save(global_model, f'{file_name}.pt')
    print(f'Training took {time_elapsed_secs:.2f} seconds / {time_elapsed_mins:.2f} minutes')
    
    
   

    
    
    
      
              