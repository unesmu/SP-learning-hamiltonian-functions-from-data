import numpy as np
import random
import torch
import json
import dill as pickle

def count_parameters(model):
    '''
    from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_device():
    # set device to GPU if available otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # if there is a GPU
        print (f'Available device : {torch.cuda.get_device_name(0)}') 
    else :
        print('Available device:',device)
    return device

def save_stats(stats, stats_path):
    with open(stats_path, 'w') as file:
        file.write(json.dumps(stats))  
    return

def set_all_seeds(manualSeed = 123, new_results=False):
    # Set random seed for reproducibility
    if new_results:
        manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

def read_dict(stats_path):
    # read the stats txt file
    with open(stats_path) as f:
        data = f.read()
    data = json.loads(data)
    return data

def pickle_save(file, path):
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    with open(path, 'rb') as handle:
        file = pickle.load(handle)
    return file