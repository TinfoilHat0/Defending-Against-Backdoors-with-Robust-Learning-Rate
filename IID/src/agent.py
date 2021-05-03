import torch
import models
import utils
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import OrderedDict



class Agent():
    def __init__(self, id, args, train_dataset, data_idxs, criterion, writer):
        self.id = id
        self.args = args
        self.criterion = criterion
        self.writer = writer 
        self.model = models.get_model(args.data).to(args.device)
        # for backdoor attack, agent poisons his local dataset
        if args.attack == 2 and self.id < args.num_corrupt:
            utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
            
        self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                
        args.nesterov = False if self.args.client_moment == 0 else self.args.nesterov
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.client_lr,\
            momentum=args.client_moment, weight_decay=args.client_wd, nesterov=args.nesterov)
        
        
    def local_train(self, global_model, cur_round):
        """ Do a local training over the received global model, return the update """
        if self.args.projected_gd:
            global_model_params = parameters_to_vector(global_model.parameters())        
            
        vector_to_parameters(parameters_to_vector(global_model.parameters()), self.model.parameters())
        self.model.train()      
        train_loss, train_acc = 0, 0
        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                labels.to(device=self.args.device, non_blocking=True)
                                
                outputs = self.model(inputs)
                minibatch_loss = self.criterion(outputs, labels)
                minibatch_loss.backward()
                
                nn.utils.clip_grad_norm_(self.model.parameters(), 10) # to prevent exploding gradients
                self.optimizer.step()

                with torch.no_grad():
                    # doing projected gradient descent to ensure the update is within the norm bounds of server
                    if  self.args.clip > 0 and self.args.projected_gd:
                        local_model_params = parameters_to_vector(self.model.parameters())
                        param_diff = local_model_params - global_model_params
                        clip_denom = max(1, torch.norm(param_diff, p=2)/self.args.clip)
                        param_diff.div_(clip_denom)
                        vector_to_parameters(global_model_params + param_diff, self.model.parameters())
                        
                    # train_loss += minibatch_loss.item()*outputs.shape[0]
                    # _, pred_labels = torch.max(outputs, 1)=
                    # train_acc += torch.sum(torch.eq(pred_labels.view(-1), labels)).item()
                           
                           
        with torch.no_grad():
            # train_loss, train_acc = train_loss/(len(self.train_dataset)*self.args.local_ep),\
            #      train_acc/(len(self.train_dataset)*self.args.local_ep)
            # self.writer.add_scalar(f'Agents/Loss/Agent_{self.id}', train_loss, cur_round)
            # self.writer.add_scalar(f'Agents/Accuracy/Agent_{self.id}', train_acc, cur_round)
            
            update = parameters_to_vector(self.model.parameters()).double() \
                    -parameters_to_vector(global_model.parameters())        
            return update
            
