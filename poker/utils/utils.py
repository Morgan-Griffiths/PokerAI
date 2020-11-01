import numpy as np
import os
import pickle
from torch import where,zeros_like
import re

def bin_by_handstrength(strength):
    if strength > 6185:
        return 8
    if strength > 3325:
        return 7
    if strength > 2467:
        return 6
    if strength > 1609:
        return 5
    if strength > 1599:
        return 4
    if strength > 322:
        return 3
    if strength > 166:
        return 2
    if strength > 10:
        return 1
    return 0

def grep(pat, txt): 
    r = re.search(pat, txt)
    return r.group(0) if r else 1e+10

def return_next_baseline_path(path):
    baselines_path = return_latest_baseline_path(path)
    if baselines_path:
        max_num = path.rsplit('baseline')[-1]
        return os.path.join(path,f'baseline{int(max_num)+1}')
    else:
        return os.path.join(path,'baseline1')

def return_latest_baseline_path(path):
    baselines_paths = load_paths(path)
    if baselines_paths:
        agents = {}
        highest_number = 0
        for name,b_path in baselines_paths.items():
            number = int(name.split('baseline')[-1])
            agents[number] = b_path
            highest_number = max(highest_number,number)
        return agents[highest_number]
    return ''

def load_paths(folder):
    weight_paths = {}
    for weight_file in os.listdir(folder):
        if weight_file != '.DS_Store' and not os.path.isdir(os.path.join(folder,weight_file)):
            weight_paths[weight_file] = os.path.join(folder,weight_file)
    return weight_paths

def clean_folder(folder):
    try:
        for weight_file in os.listdir(folder):
            if weight_file != '.DS_Store':
                os.remove(os.path.join(folder,weight_file))
    except:
        pass

def return_uniques(values):
    uniques, count = np.unique(values, return_counts=True)
    return uniques, count

def torch_where(condition,vec):
    mask = where(condition,vec,zeros_like(vec))
    mask[mask>0]= 1
    return mask

def savepickle(data,path):
    #store data
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)   
        
def openpickle(path):
    #to read file
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b

def unpack_shared_dict(shared):
    combined_dict = {position: [] for position in ['SB','BB']}
    keys = shared.keys()
    for position in ['SB','BB']:
        # for keyo in keys:
        #     print(len(shared[keyo][position]))
        for key in keys:
            for i in range(len(shared[key][position])):
                try:
                    combined_dict[position].append(shared[key][position][i])
                except:
                    pass
    return combined_dict