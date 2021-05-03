import torch
import utils
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import torch.nn as nn
import random
from torch.distributions.normal import Normal
from torch.nn import functional as F
from copy import deepcopy
import math
import copy
import numpy as np


class Aggregation():
    def __init__(self, global_model, poisoned_val_loader, args, writer):
        self.args = args
        self.server_lr = args.server_lr
        self.global_model = global_model
        self.writer = writer
        self.poisoned_val_loader = poisoned_val_loader
        self.n_params = len(parameters_to_vector(global_model.parameters()))
        self.normal_dist = Normal(0, args.noise*args.clip)
        self.cum_net_mov = 0
        
        
    def aggregate_updates(self, agent_updates, cur_round):
        cur_global_params = parameters_to_vector(self.global_model.parameters())
        
        # plotting some statistics regarding agent updates
        self.plot_norms(agent_updates, cur_round)
        
        # adjust LR if robust LR is selected
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.robust_lr > 0:
            lr_vector = self.compute_robustLR(agent_updates)
            
              
                   
        aggregated_updates = 0
        if self.args.aggr == 'avg':
            aggregated_updates = self.agg_avg(agent_updates)
        elif self.args.aggr == 'comed':
            aggregated_updates = self.agg_comed(agent_updates)     
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates)
            
        if self.args.noise > 0:
            aggregated_updates.add_(self.normal_dist.sample( (self.n_params,) ).to(self.args.device))
        
        new_global_params = cur_global_params + lr_vector*(aggregated_updates - self.args.server_wd*cur_global_params) 
        vector_to_parameters(new_global_params.float(), self.global_model.parameters())
        
        #self.plot_sign_agreement(lr_vector, cur_global_params, new_global_params, cur_round)     
        return           
    
     
    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [torch.sign(update) for update in agent_updates]  
        sm_of_signs, adjusted_lr = torch.abs(sum(agent_updates_sign)), 0
        if self.args.robust_lr == 1:
            sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = 0 
            sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr
             
        elif self.args.robust_lr == 2:
            sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
            sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr                                           
        return sm_of_signs.to(self.args.device)
    
            
    def agg_avg(self, agent_updates):
        """ classic fed avg """        
        return sum(agent_updates) / self.args.num_agents
    
    
    def agg_sign(self, agent_updates):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    
    def agg_comed(self, agent_updates):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
       
    def clip_updates(self, agent_updates):
        for update in agent_updates:
            l2_update = torch.norm(update, p=2) # get l2 of update
            update.div_(max(1, l2_update/self.args.clip))
        return
    
                   
    def add_noise_to_updates(self, agent_updates, lr_vector, noise):
        for update in agent_updates:
            noise = utils.compute_noise(update, noise).to(self.args.device)
            update.add_(lr_vector*noise)
        return
    
       
    def null_attack(self, agent_updates):
        """ flip updates of corrupt agents """
        for i in range(self.args.num_corrupt):
            agent_updates[i] *= 0
        return agent_updates
    
       
    def flip_signs(self, agent_updates):
        """ flip updates of corrupt agents """
        for i in range(self.args.num_corrupt):
                agent_updates[i] *= -1
        return agent_updates
    
       
    def boost_updates(self, agent_updates):
        """ boost update of corrupt agents """
        for i in range(self.args.num_corrupt):
            agent_updates[i] *= (self.args.num_agents - self.args.num_corrupt+1) / self.args.num_corrupt
        return agent_updates     
    
    
    def comp_diag_fisher(self, model_params, data_loader, adv=True):
    
        model = models.get_model(self.args.data)
        vector_to_parameters(model_params, model.parameters())
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data
            
        model.eval()
        for _, (inputs, labels) in enumerate(data_loader):
            model.zero_grad()
            inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                    labels.to(device=self.args.device, non_blocking=True).view(-1, 1)
            if not adv:
                labels.fill_(self.args.base_class)
                
            outputs = model(inputs)
            log_all_probs = F.log_softmax(outputs, dim=1)
            target_log_probs = outputs.gather(1, labels)
            batch_target_log_probs = target_log_probs.sum()
            batch_target_log_probs.backward()
            
            for n, p in model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)
                
        return parameters_to_vector(precision_matrices.values()).detach()
        
        
    def plot_norms(self, agent_updates, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        l2_honest_updates = [torch.norm(update, p=norm) for update in agent_updates[self.args.num_corrupt:]]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)
        
        if self.args.num_corrupt > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in agent_updates[:self.args.num_corrupt]]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round) 
        return
    
    
    def plot_sign_agreement(self, robustLR, cur_global_params, new_global_params, cur_round):
        """ Getting sign agreement of updates between honest and corrupt agents """
        # total update for this round
        update = new_global_params - cur_global_params
        
        # compute FIM to quantify these parameters: (i) parameters which induces adversarial mapping on trojaned, (ii) parameters which induces correct mapping on trojaned
        fisher_adv = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader)
        fisher_hon = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader, adv=False)
        _, adv_idxs = fisher_adv.sort()
        _, hon_idxs = fisher_hon.sort()
        
        # get most important n_idxs params
        n_idxs = self.args.top_frac #math.floor(self.n_params*self.args.top_frac)
        adv_top_idxs = adv_idxs[-n_idxs:].cpu().detach().numpy()
        hon_top_idxs = hon_idxs[-n_idxs:].cpu().detach().numpy()
        
        # minimized and maximized indexes
        min_idxs = (robustLR == -self.args.server_lr).nonzero().cpu().detach().numpy()
        max_idxs = (robustLR == self.args.server_lr).nonzero().cpu().detach().numpy()
        
        # get minimized and maximized idxs for adversary and honest
        max_adv_idxs = np.intersect1d(adv_top_idxs, max_idxs)
        max_hon_idxs = np.intersect1d(hon_top_idxs, max_idxs)
        min_adv_idxs = np.intersect1d(adv_top_idxs, min_idxs)
        min_hon_idxs = np.intersect1d(hon_top_idxs, min_idxs)
       
        # get differences
        max_adv_only_idxs = np.setdiff1d(max_adv_idxs, max_hon_idxs)
        max_hon_only_idxs = np.setdiff1d(max_hon_idxs, max_adv_idxs)
        min_adv_only_idxs = np.setdiff1d(min_adv_idxs, min_hon_idxs)
        min_hon_only_idxs = np.setdiff1d(min_hon_idxs, min_adv_idxs)
        
        # get actual update values and compute L2 norm
        max_adv_only_upd = update[max_adv_only_idxs] # S1
        max_hon_only_upd = update[max_hon_only_idxs] # S2
        
        min_adv_only_upd = update[min_adv_only_idxs] # S3
        min_hon_only_upd = update[min_hon_only_idxs] # S4


        #log l2 of updates
        max_adv_only_upd_l2 = torch.norm(max_adv_only_upd).item()
        max_hon_only_upd_l2 = torch.norm(max_hon_only_upd).item()
        min_adv_only_upd_l2 = torch.norm(min_adv_only_upd).item()
        min_hon_only_upd_l2 = torch.norm(min_hon_only_upd).item()
       
        self.writer.add_scalar(f'Sign/Hon_Maxim_L2', max_hon_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Maxim_L2', max_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Minim_L2', min_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Minim_L2', min_hon_only_upd_l2, cur_round)
        
        
        net_adv =  max_adv_only_upd_l2 - min_adv_only_upd_l2
        net_hon =  max_hon_only_upd_l2 - min_hon_only_upd_l2
        self.writer.add_scalar(f'Sign/Adv_Net_L2', net_adv, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Net_L2', net_hon, cur_round)
        
        self.cum_net_mov += (net_hon - net_adv)
        self.writer.add_scalar(f'Sign/Model_Net_L2_Cumulative', self.cum_net_mov, cur_round)
        
        
        
        """
        # get idxs differences and conjunction between adv. and honest params.
        inter_idxs = np.intersect1d(adv_top_idxs, hon_top_idxs)
        adv_diff_idxs = np.setdiff1d(adv_top_idxs, hon_top_idxs)
        hon_diff_idxs = np.setdiff1d(hon_top_idxs, adv_top_idxs)
        
        
        # get params for l2s
        inter_params = global_params[inter_idxs]
        adv_diff_params = global_params[adv_diff_idxs]
        hon_diff_params = global_params[hon_diff_idxs]
        
        # log l2s
        inter_l2 = torch.norm(inter_params).item()
        adv_diff_l2 = torch.norm(adv_diff_params).item()
        hon_diff_l2 = torch.norm(hon_diff_params).item()
        self.writer.add_scalar(f'Sign/Intersection_L2', inter_l2, cur_round)
        self.writer.add_scalar(f'Sign/Hon_L2', hon_diff_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_L2', adv_diff_l2, cur_round)
        
        

        # log counts for idxs
        self.writer.add_scalar(f'Sign/Intersection_Count', len(inter_idxs) , cur_round)
    
       
        # get indexes of adv, intersection, hon maximimized and their L2 norms
        min_inter_idxs = np.intersect1d(inter_idxs, min_idxs)
        min_adv_diff_idxs = np.intersect1d(adv_diff_idxs, min_idxs)
        min_hon_diff_idxs = np.intersect1d(hon_diff_idxs, min_idxs)
        min_rand_idxs = np.intersect1d(rand_idxs, min_idxs)
        
        self.writer.add_scalar(f'Sign/Minimized_Intersection_Count', len(min_inter_idxs), cur_round)
        self.writer.add_scalar(f'Sign/Minimized_AdvDiff_Count', len(min_adv_diff_idxs), cur_round)
        self.writer.add_scalar(f'Sign/Minimized_HonDiff_Count', len(min_hon_diff_idxs), cur_round)
        self.writer.add_scalar(f'Sign/Minimized_Rand_Count', len(min_rand_idxs), cur_round)
        
         # get params for l2s
        min_inter_params = global_params[min_inter_idxs]
        min_adv_diff_params = global_params[min_adv_diff_idxs]
        min_hon_diff_params = global_params[min_hon_diff_idxs]
        
        min_inter_l2 = torch.norm(min_inter_params).item()
        min_adv_diff_l2 = torch.norm(min_adv_diff_params).item()
        min_hon_diff_l2 = torch.norm(min_hon_diff_params).item()
        
        self.writer.add_scalar(f'Sign/Intersection_L2', min_inter_l2, cur_round)
        self.writer.add_scalar(f'Sign/AdvDiff_L2', min_adv_diff_l2, cur_round)
        self.writer.add_scalar(f'Sign/HonDiff_L2', min_hon_diff_l2, cur_round)
        
        
         # get actual parameters from computed indexes for l2 norm
        inter_params = global_params[inter_idxs]
        adv_diff_params = global_params[adv_diff_idxs]
        hon_diff_params = global_params[hon_diff_idxs]
        
        inter_l2 = torch.norm(inter_params).item()
        adv_diff_l2 = torch.norm(adv_diff_params).item()
        hon_diff_l2 = torch.norm(hon_diff_params).item()
        self.writer.add_scalar(f'Sign/Intersection_L2', inter_l2, cur_round)
        self.writer.add_scalar(f'Sign/AdvDiff_L2', adv_diff_l2, cur_round)
        self.writer.add_scalar(f'Sign/HonDiff_L2', hon_diff_l2, cur_round)
        """
        
        return
                