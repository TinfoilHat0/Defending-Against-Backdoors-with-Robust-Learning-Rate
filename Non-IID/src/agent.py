import torch
import models
import utils
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import OrderedDict



class Agent():
    def __init__(self, id, args,  writer):
        self.id = id
        self.args = args
        self.writer = writer
        
        self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')
        if args.attack == 2 and self.id < args.num_corrupt:
            utils.poison_dataset(self.train_dataset, args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, pin_memory=False)
        self.n_data = len(self.train_dataset)    
     
     
    def local_train(self, global_model, criterion, cur_round):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, 
            momentum=self.args.client_moment, weight_decay=self.args.client_wd, nesterov=self.args.nesterov)
        
        train_loss, train_acc = 0, 0
        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                 labels.to(device=self.args.device, non_blocking=True)
                                               
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                
                nn.utils.clip_grad_norm_(global_model.parameters(), 10) # to prevent exploding gradients
                optimizer.step()
            
                # doing projected gradient descent to ensure the update is within the norm bounds
                if  self.args.projected_gd:
                    with torch.no_grad():
                        local_model_params = parameters_to_vector(global_model.parameters())
                        update = local_model_params - initial_global_model_params
                        clip_denom = max(1, torch.norm(update, p=2)/self.args.clip)
                        update.div_(clip_denom)
                        vector_to_parameters(initial_global_model_params + update, global_model.parameters())
                            
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update
            
