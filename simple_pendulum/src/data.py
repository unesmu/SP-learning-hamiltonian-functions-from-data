import torch 
from torch.utils.data import Dataset, DataLoader, random_split


class TrajectoryDataset(Dataset):
    '''
    
    '''
    def __init__(self, device, q, p, t_eval):
        self.q = q.to(device)
        self.p = p.to(device)
        self.t_eval = t_eval.to(device)

    def __len__(self):
        return self.q.shape[1]

    def __getitem__(self, idx):
        q = self.q[:,idx]
        p = self.p[:,idx]
        t_eval = self.t_eval
        x = torch.stack((q,p),dim=1)
        return x, t_eval

def data_loader(q, p, t_eval, batch_size, device, shuffle = True, proportion = 0.5):
    '''
    
    '''
    # split  into train and test 
    if proportion:

        full_dataset = TrajectoryDataset(device, q, p, t_eval)

        train_size = int(proportion * len(full_dataset))
        test_size = len(full_dataset) - train_size

        train, test = random_split(full_dataset, [train_size, test_size])

        test_loader = DataLoader(
            test,
            batch_size,
            shuffle
            )    
    else:
      train = TrajectoryDataset(device, q, p, t_eval)
      # if proportion is set to None don't split the dataset
      q_train = q
      p_train = p
      t_eval_train = t_eval
      test_loader = None

    # load the training data into a custom pyotorch dataset

    # create the dataloader object from the custom dataset
    train_loader = DataLoader(
        train,
        batch_size,
        shuffle
        )
    
    return train_loader, test_loader