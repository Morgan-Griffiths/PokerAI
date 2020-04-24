import numpy as np
import random
import torch
from collections import namedtuple, deque
import math

"""
Priority Tree.
3 tiered tree structure containing
Root node (Object. sum of all lower values)
Intermediate Node (Object. Root as parent, sums a given slice of the priority array)
Priority Array (Array of priorities, length buffer_size)

The number of Intermediate nodes is calculated by the buffer_size / batch_size.

I_episode: current episode of training

Index: is calculated by i_episode % buffer_size. This loops the index after exceeding the buffer_size.

Indicies: (List) of memory/priority entries

intermediate_dict: maps index to intermediate node. Since each Intermediate node is responsible 
for a given slice of the priority array, given a particular index, it will return the Intermediate node
'responsible' for that index.

## Functions:

Add:
Calculates the priority of each TD error -> (abs(TD_error)+epsilon)**alpha
Stores the priority in the Priority_array.
Updates the sum_tree with the new priority

Update_Priorities:
Updates the index with the latest priority of that sample. As priorities can change over training
for a particular experience

Sample:
Splits the current priority_array based on the number of entries, by the batch_size.
Returns the indicies of those samples and the priorities.

Propogate:
Propogates the new priority value up through the tree
"""

class PriorityTree(object):
    def __init__(self,buffer_size,batch_size,alpha,epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_intermediate_nodes = math.ceil(buffer_size / batch_size)
        self.current_intermediate_node = 0
        self.root = Node(None)
        self.intermediate_nodes = [Intermediate(self.root,batch_size*x,batch_size*(x+1)) for x in range(self.num_intermediate_nodes)]
        self.priority_array = np.zeros(buffer_size)
        self.intermediate_dict = {}
        for index,node in enumerate(self.intermediate_nodes):
            for key in range((batch_size*(index+1))-batch_size,batch_size*(index+1)):
                self.intermediate_dict[key] = node
        print('Priority Tree: Batch Size {} Buffer size {} Number of intermediate Nodes {}'.format(batch_size,buffer_size,self.num_intermediate_nodes))
        
    def add(self,TD_error,index):
        priority = (abs(TD_error)+self.epsilon)**self.alpha
        self.priority_array[index] = priority
        # Update sum
        propogate(self.intermediate_dict[index],self.priority_array)
    
    def sample(self,index):
        # Sample one experience uniformly from each slice of the priorities
        if index >= self.buffer_size:
            indicies = [random.sample(list(range(sample*self.num_intermediate_nodes,(sample+1)*self.num_intermediate_nodes)),1)[0] for sample in range(self.batch_size)]
        else:
            interval = int(index / self.batch_size)
            indicies = [random.sample(list(range(sample*interval,(sample+1)*interval)),1)[0] for sample in range(self.batch_size)]
#         print('indicies',indicies)
        priorities = self.priority_array[indicies]
        return priorities,indicies
    
    def update_priorities(self,TD_errors,indicies):
#         print('TD_errors',TD_errors)
#         print('TD_errors shape',TD_errors.shape)
        priorities = (abs(TD_errors)+self.epsilon)**self.alpha
#         print('priorities shape',priorities.shape)
#         print('indicies shape',len(indicies))
#         print('self.priority_array shape',self.priority_array.shape)
        self.priority_array[indicies] = priorities
        # Update sum
        nodes = [self.intermediate_dict[index] for index in indicies] 
        intermediate_nodes = set(nodes)
        [propogate(node,self.priority_array) for node in intermediate_nodes]
    
class Node(object):
    def __init__(self,parent):
        self.parent = parent
        self.children = []
        self.value = 0
            
    def add_child(self,child):
        self.children.append(child)
    
    def set_value(self,value):
        self.value = value
    
    def sum_children(self):
        return sum([child.value for child in self.children])
            
    def __len__(self):
        return len(self.children)

class Intermediate(Node):
    def __init__(self,parent,start,end):
        self.parent = parent
        self.start = start
        self.end = end
        self.value = 0
        parent.add_child(self)
    
    def sum_leafs(self,arr):
        return np.sum(arr[self.start:self.end])

def propogate(node,arr):
    if node.parent != None:
        node.value = node.sum_leafs(arr)
        propogate(node.parent,arr)
    else:
        node.value = node.sum_children()

"""
Priority Buffer HyperParameters
alpha(priority or w) dictates how biased the sampling should be towards the TD error. 0 < a < 1
beta(IS) informs the importance of the sample update

The paper uses a sum tree to calculate the priority sum in O(log n) time. As such, i've implemented my own version
of the sum_tree which i call priority tree.

We're increasing beta(IS) from 0.5 to 1 over time
alpha(priority) we're holding constant at 0.5
"""

class PriorityReplayBuffer(object):
    def __init__(self,action_size,buffer_size,batch_size,seed,alpha=0.5,beta=0.5,beta_end=1,beta_duration=1e+5,epsilon=7e-5):
        
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_end = beta_end
        self.beta_duration = beta_duration
        self.beta_increment = (beta_end - beta) / beta_duration
        self.max_w = 0
        self.epsilon = epsilon
        self.TD_sum = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.experience = namedtuple('experience',field_names=['state','action','reward','next_state','done','i_episode'])
        self.sum_tree = PriorityTree(buffer_size,batch_size,alpha,epsilon)
        self.memory = {}
    
    def add(self,state,action,reward,next_state,done,TD_error,i_episode):
        e = self.experience(state,action,reward,next_state,done,i_episode)
        index = i_episode % self.buffer_size
        # add memory to memory and add corresponding priority to the priority tree
        self.memory[index] = e
        self.sum_tree.add(TD_error,index)

    def sample(self,index):
        # We times the error by these weights for the updates
        # Super inefficient to sum everytime. We could implement the tree sum structure. 
        # Or we could sum once on the first sample and then keep track of what we add and lose from the buffer.
        # priority^a over the sum of the priorities^a = likelyhood of the given choice
        # Anneal beta
        self.update_beta()
        # Get the samples and indicies
        priorities,indicies = self.sum_tree.sample(index)
        # Normalize with the sum
        norm_priorities = priorities / self.sum_tree.root.value
        samples = [self.memory[index] for index in indicies]
#         samples = list(operator.itemgetter(*self.memory)(indicies))
#         samples = self.memory[indicies]
        # Importance weights
#         print('self.beta',self.beta)
#         print('self.beta',self.buffer_size)
        importances = [(priority * self.buffer_size)**-self.beta for priority in norm_priorities]
        self.max_w = max(self.max_w,max(importances))
        # Normalize importance weights
#         print('importances',importances)
#         print('self.max_w',self.max_w)
        norm_importances = [importance / self.max_w for importance in importances]
#         print('norm_importances',norm_importances)
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(self.device)
#         np.vstack([e.done for e in samples if e is not None]).astype(int)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(int)).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None])).float().to(self.device)
        
        if index % 4900 == 0:
            print('beta',self.beta)
            print('self.max_w',self.max_w)
            print('len mem',len(self.memory))
            print('tree sum',self.sum_tree.root.value)
        
        return (states,actions,rewards,next_states,dones),indicies,norm_importances

    def update_beta(self):
#         print('update_beta')
#         print('self.beta_end',self.beta_end)
#         print('self.beta_increment',self.beta_increment)
        self.beta += self.beta_increment
        self.beta = min(self.beta,self.beta_end)
    
    def __len__(self):
        return len(self.memory.keys())