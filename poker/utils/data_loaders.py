import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

class trajectoryLoader(Dataset):
    def __init__(self, data):
        betsize_masks = []
        action_masks = []
        actions = []
        states = []
        rewards = []
        for i,poker_round in enumerate(data):
            states.append(torch.tensor(poker_round['state'],dtype=torch.float32).permute(1,0,2))
            actions.append(torch.tensor(poker_round['action'],dtype=torch.long))
            rewards.append(torch.tensor(poker_round['reward'],dtype=torch.float32))
            betsize_masks.append(torch.tensor(poker_round['betsize_mask'],dtype=torch.long))
            action_masks.append(torch.tensor(poker_round['action_mask'],dtype=torch.long))
        self.states = pad_sequence(states).squeeze(2).permute(1,0,2)
        self.actions = torch.stack(actions)#.unsqueeze(-1)
        self.action_masks = torch.stack(action_masks)
        self.betsize_masks = torch.stack(betsize_masks)
        self.rewards = torch.stack(rewards)#.unsqueeze(-1)
        print(f'states:{self.states.size()},actions:{self.actions.size()},action_masks:{self.action_masks.size()},betsize_masks:{self.betsize_masks.size()},rewards:{self.rewards.size()}')

    def __len__(self):
        return self.states.size()[0]

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        sample = {'X':{'state': self.states[idx],'action':self.actions[idx],'betsize_mask':self.betsize_masks[idx],'action_mask':self.action_masks[idx]},'y':{ 'reward': self.rewards[idx]}}
        return sample

def return_trajectoryloader(data):
    data = trajectoryLoader(data)
    params = {
        'batch_size':48,
        'shuffle': True,
        'num_workers':2
    }
    trainloader = DataLoader(data, **params)
    return trainloader