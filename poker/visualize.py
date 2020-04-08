
import matplotlib.pyplot as plt
from db import MongoDB

hand_dict = {0:'?',1:'Q',2:'K',3:'A'}

label_dict = {5:['Checking','Betting'],
            0:['Check','Bet'],
            1:['Call','Raise','Fold'],
            4:['Betting','Checking']}

action_labels = {
    0:'check',
    1:'bet',
    2:'call',
    3:'fold',
    4:'raise'
}

colors = ['g','b','m','r']

def visualize_actions(actions):
    labels = label_dict[index]
    keys = log_probs.keys()
    N = 0
    for key in keys:
        N = max(N,len(log_probs[key]))
    epochs = range(1,N+1)
    for i in range(len(keys)):
        plt.plot(epochs,torch.exp(log_probs[i]).detach().numpy(),colors[i],label=f"{action_labels[i]} frequency")
#     plt.title(f'{card_dict[hand]}')
    plt.xlabel('Epochs')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('DQN_performance.png',bbox_inches='tight')

def monitor_frequencies(hand,log_probs,index):
    labels = label_dict[index]
    keys = log_probs.keys()
    N = 0
    for key in keys:
        N = max(N,len(log_probs[key]))
    epochs = range(1,N+1)
    for i in range(len(keys)):
        plt.plot(epochs,torch.exp(log_probs[i]).detach().numpy(),colors[i],label=f"{action_labels[i]} frequency")
#     plt.title(f'{card_dict[hand]}')
    plt.xlabel('Epochs')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('DQN_performance.png',bbox_inches='tight')
    
def monitor_ndfrequencies(hand,log_probs,index):
    labels = label_dict[index]
    epochs = range(1,log_probs.size(0)+1)
    for i in range(log_probs.size(1)):
        print(log_probs[:,i].detach().numpy().size)
        plt.plot(epochs,log_probs[:,i].detach().numpy(),colors[i],label=f"{action_labels[i]} frequency")
#     plt.title(f'{card_dict[hand]}')
    plt.xlabel('Epochs')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('DQN_performance.png',bbox_inches='tight')


def plot_data(title:str,data:list,labels:list,path='assets/'):
    epochs = range(1,len(data[0])+1)
    for i,data_group in enumerate(data):
        print(data_group,i)
        plt.plot(epochs,data_group,colors[i],label=labels[i])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{path+title}.png',bbox_inches='tight')
    plt.close()

def plot_frequencies(title:str,data:list,labels:list,path='assets/'):
    print(f'data dimensions: {len(data[0])}, {len(data[0][0])}')
    M = len(data[0][0])
    amount = M
    epochs = range(amount)#range(1,len(data[0][0])+1)
    fig, axs = plt.subplots(len(data))
    fig.suptitle('Frequencies')
    barWidth = 0.5
    for i,hand in enumerate(data):
        axs[i].bar(epochs,hand[0][:amount],color=colors[0],label=labels[0],width=barWidth)
        axs[i].bar(epochs,hand[1][:amount],bottom=hand[0][:amount],color=colors[1],label=labels[1], width=barWidth)
        axs[i].bar(epochs,hand[2][:amount],bottom=[i+j for i,j in zip(hand[0][:amount], hand[1][:amount])],color=colors[2],label=labels[2],width=barWidth)
        axs[i].grid(True)
        axs[i].set_title(f'Hand {i}')
        # axs[i].set_xlabel('Epochs')
        # axs[i].set_ylabel('Frequency')
        # axs[i].legend()
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(f'{path+title}.png',bbox_inches='tight')
    # plt.title(title)
    # plt.xlabel('Epochs')
    # plt.ylabel('Frequency')
    # plt.legend()
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Visualize training runs\n\n
        """)

    parser.add_argument('--run',
                        default=0,
                        metavar="integer",
                        help='Which training run to look at')
    parser.add_argument('--round',
                        default=None,
                        metavar="[integer >= 0,None]",
                        help='Which round within each training run')
    parser.add_argument('--step',
                        default=0,
                        metavar="integer",
                        help='Which step within each round')
    parser.add_argument('--position',
                        default='SB',
                        metavar="['SB','BB']",
                        help='Which position to look at')
    parser.add_argument('--type',
                        default='action',
                        metavar="['game_state,observation,action,reward']",
                        help='Chooses the type of data to look at')

    args = parser.parse_args()
    
    query = {
        'position':args.position,
        # 'type':args.type,
        # 'step':0,
        # 'poker_round':0,
        'training_round':args.run
    }

    projection ={'action':1,'hand':1,'_id':0}
    params = {
        'interval':100,
        'num_features':3
    }
    # projection = None
    mongo = MongoDB()
    print(query)
    """
    Plots: 
    Rewards over time. Action frequencies over time. Action frequencies given hand over time.
    Action frequencies given state over time.
    """
    # actions = mongo.byActions(query,action_only=True)
    # rewards = mongo.byRewards(query)
    # actions,hands = mongo.actionByHand(query)
    data = mongo.get_data(query,projection)
    actions,hands = mongo.actionByHand(data,params)
    # projection ={'action':1,'_id':0}
    # projection ={'reward':1,'_id':0}
    hand_labels = [f'Hand {hand_dict[hand]}' for hand in hands]
    labels = [f'Action {i}' for i in hands]
    print(hand_labels)
    print(labels)
    print(hands)
    # states = mongo.byStates(query)
    plot_frequencies(f'Actions probabilities for {query["position"]}',actions,labels)
    # plot_data(f'Rewards for {query["position"]}',[rewards],['Rewards'])