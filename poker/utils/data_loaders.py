import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

def pad_seq(seqs,maxlen):
    """pads over the sequence dimension"""
    shape = list(seqs[0].size())
    new_seqs = []
    for seq in seqs:
        padding_len = maxlen - seq.size(1)
        shape[1] = padding_len
        if padding_len > 0:
            padding = torch.zeros(shape,dtype=torch.float32)
            new_seqs.append(torch.cat((padding,seq),dim=1))
        else:
            new_seqs.append(seq)
    return torch.stack(new_seqs).squeeze(1)

class trajectoryLoader(Dataset):
    def __init__(self, data):
        betsize_masks = []
        action_masks = []
        actions = []
        states = []
        obs = []
        rewards = []
        maxlen = 0
        for i,poker_round in enumerate(data):
            states.append(torch.tensor(poker_round['state'],dtype=torch.float32))#.permute(1,0,2))
            obs.append(torch.tensor(poker_round['obs'],dtype=torch.float32))#.permute(1,0,2))
            actions.append(torch.tensor(poker_round['action'],dtype=torch.long))
            rewards.append(torch.tensor(poker_round['reward'],dtype=torch.float32))
            betsize_masks.append(torch.tensor(poker_round['betsize_mask'],dtype=torch.long))
            action_masks.append(torch.tensor(poker_round['action_mask'],dtype=torch.long))
            maxlen = max(maxlen,torch.tensor(poker_round['state']).size(1))
        self.states = pad_seq(states,maxlen)#.squeeze(2).permute(1,0,2)
        self.obs = pad_seq(obs,maxlen)#.squeeze(2).permute(1,0,2)
        self.actions = torch.stack(actions)#.unsqueeze(-1)
        self.action_masks = torch.stack(action_masks)
        self.betsize_masks = torch.stack(betsize_masks)
        self.rewards = torch.stack(rewards)#.unsqueeze(-1)
        # print(f'states:{self.states.size()},obs:{self.obs.size()},actions:{self.actions.size()},action_masks:{self.action_masks.size()},betsize_masks:{self.betsize_masks.size()},rewards:{self.rewards.size()}')

    def __len__(self):
        return self.states.size()[0]

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        sample = {'X':{'state': self.states[idx],'obs':self.obs[idx],'action':self.actions[idx],'betsize_mask':self.betsize_masks[idx],'action_mask':self.action_masks[idx]},'y':{ 'reward': self.rewards[idx]}}
        return sample

def return_trajectoryloader(data):
    data = trajectoryLoader(data)
    params = {
        'batch_size':128,
        'shuffle': True,
        'num_workers':4
    }
    trainloader = DataLoader(data, **params)
    return trainloader