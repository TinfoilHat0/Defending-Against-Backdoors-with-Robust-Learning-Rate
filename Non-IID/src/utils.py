import torch
import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from math import floor, sqrt, pi
from collections import defaultdict, Counter
from torch.distributions.normal import Normal
import random
import h5py
import torch.nn as nn
import cv2


class H5Dataset(Dataset):
    def __init__(self, dataset, client_id):
        self.targets = torch.LongTensor(dataset[client_id]['label'])
        self.inputs = torch.Tensor(dataset[client_id]['pixels'])
        shape = self.inputs.shape
        self.inputs = self.inputs.view(shape[0], 1, shape[1], shape[2])
        
    def classes(self):
        return torch.unique(self.targets)
    
    def __add__(self, other): 
        self.targets = torch.cat( (self.targets, other.targets), 0)
        self.inputs = torch.cat( (self.inputs, other.inputs), 0)
        return self
    
    def to(self, device):
        self.targets = self.targets.to(device)
        self.inputs = self.inputs.to(device)
        
        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, item):
        inp, target = self.inputs[item], self.targets[item]
        return inp, target


class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        
    def classes(self):
        return torch.unique(self.targets)    

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target



def distribute_data(dataset, args, n_classes=10):
    if args.num_agents == 1:
        return {0:range(len(dataset))}
    
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
        
    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * args.class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size    
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
           
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        for j in range(user_idx, user_idx+args.class_per_agent):
            dict_users[user_idx] += labels_dict[j%n_classes][0]
            del labels_dict[j%n_classes][0]

    return dict_users       

  
def get_datasets(data, augment=True):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = '../data'
    
    if data == 'fmnist':
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)   

    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir)
        
    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  
        
    return train_dataset, test_dataset    

   
def compute_empirical_pmf(dataset):
    '''
    Given a dataset, returns its empirical pmf
    '''
    d = Counter(dataset.targets.tolist())
    for k in d.keys():
        d[k] /= len(dataset)
    return d    

            
def compute_avg_emd_of_data_split(train_dataset, agents):
    '''
    Computes the avg emd of the data split across agents
    wrt to the initial distribution of the dataset
    '''
    avg_emd = 0
    dataset_empirical_pmf = compute_empirical_pmf(train_dataset)
    for agent in agents:
        agent_emd = 0
        for k in dataset_empirical_pmf.keys():
            agent_emd += abs(agent.empirical_pmf[k] - dataset_empirical_pmf[k])
        avg_emd += agent_emd    
    return avg_emd / len(agents)      


def chunker_list(seq, size):
    return [seq[i::size] for i in range(size)]



def poison_dataset(dataset, args, data_idxs=None, poison_all=False):
    all_idxs = (dataset.targets == args.base_class).nonzero().flatten().tolist()
    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))
        
    poison_frac = 1 if poison_all else args.poison_frac    
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))
    for idx in poison_idxs:
        clean_img = dataset.inputs[idx]
        bd_img = add_pattern_bd(clean_img, pattern_type=args.pattern_type)
        dataset.inputs[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = args.target_class    
    return


def add_pattern_bd(x, pattern_type='square'):
    """
    adds a trojan pattern to the image
    """
    x = np.array(x.squeeze())
    
    if pattern_type == 'square':
        for i in range(21, 26):
            for j in range(21, 26):
                x[i, j] = 0
    
    elif pattern_type == 'copyright':
        trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
        trojan = cv2.bitwise_not(trojan)
        trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
        x = x - trojan
        
    elif pattern_type == 'apple':
        trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
        trojan = cv2.bitwise_not(trojan)
        trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
        x = x - trojan
        
    elif pattern_type == 'plus':
        start_idx = 8
        size = 5
        # vertical line  
        for i in range(start_idx, start_idx+size):
            x[i, start_idx] = 0
        
        # horizontal line
        for i in range(start_idx-size//2, start_idx+size//2 + 1):
            x[start_idx+size//2, i] = 0
            
    return x


def get_loss_n_accuracy(model, data_loader, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
                                            
        # compute the total loss over minibatch
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)

def print_exp_details(args):
    print('======================================')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Num. of class per agent: {args.class_per_agent}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    Server_Momentum: {args.server_moment}')
    print(f'    Client_WD: {args.client_wd}')
    print(f'    Server_WD: {args.server_wd}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Attack type: {args.attack}')
    print(f'    Robust LR: {args.robust_lr}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Clip: {args.clip}')
    print(f'    Projected_GD: {args.projected_gd}')
    print('======================================')
    