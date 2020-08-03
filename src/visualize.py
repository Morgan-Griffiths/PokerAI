
import matplotlib.pyplot as plt
import numpy as np

from db import MongoDB
from poker.datatypes import ACTION_DICT
import kuhn.datatypes as pdt

label_dict = {5:['Check','Bet'],
            0:['Check','Bet'],
            1:['Call','Raise','Fold'],
            4:['Betting','Checking']}


HAND_LABELS_DICT = {
    'frequency': lambda x: [f'Hand_strength category {i}' for i,hand in enumerate(x)],
    'probability': lambda x: [f'Hand {pdt.Globals.KUHN_CARD_DICT[hand]}' for hand in x]
}

colors = ['g','b','m','r','y']

def plot_data(title:str,data:list,labels:list,path='assets/'):
    print(path+title)
    epochs = range(1,len(data[0])+1)
    for i,data_group in enumerate(data):
        plt.plot(epochs,data_group,colors[i],label=labels[i])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path+title}.png',bbox_inches='tight')
    plt.close()

def plot_frequencies(title:str,data:list,hand_labels:list,action_labels:list,path='assets/'):
    print(path+title)
    print(f'data dimensions: {len(data)}, {len(data[0])}, {len(data[0][0])}')
    M = len(data[0][0])
    amount = M
    epochs = range(amount)#range(1,len(data[0][0])+1)
    fig, axs = plt.subplots(len(data))
    fig.suptitle('Frequencies')
    barWidth = 1
    for i,hand in enumerate(data):
        if len(action_labels) == 5:
            axs[i].bar(epochs,hand[0][:amount],color=colors[0],label=action_labels[0],width=barWidth)
            axs[i].bar(epochs,hand[1][:amount],bottom=hand[0][:amount],color=colors[1],label=action_labels[1], width=barWidth)
            axs[i].bar(epochs,hand[2][:amount],bottom=[i+j for i,j in zip(hand[0][:amount], hand[1][:amount])],color=colors[2],label=action_labels[2],width=barWidth)
            axs[i].bar(epochs,hand[3][:amount],bottom=[i+j+k for i,j,k in zip(hand[0][:amount], hand[1][:amount],hand[2][:amount])],color=colors[3],label=action_labels[3],width=barWidth)
            axs[i].bar(epochs,hand[4][:amount],bottom=[i+j+k+l for i,j,k,l in zip(hand[0][:amount], hand[1][:amount],hand[2][:amount],hand[3][:amount])],color=colors[4],label=action_labels[4],width=barWidth)
        elif len(action_labels) == 4:
            axs[i].bar(epochs,hand[0][:amount],color=colors[0],label=action_labels[0],width=barWidth)
            axs[i].bar(epochs,hand[1][:amount],bottom=hand[0][:amount],color=colors[1],label=action_labels[1], width=barWidth)
            axs[i].bar(epochs,hand[2][:amount],bottom=[i+j for i,j in zip(hand[0][:amount], hand[1][:amount])],color=colors[2],label=action_labels[2],width=barWidth)
            axs[i].bar(epochs,hand[3][:amount],bottom=[i+j+k for i,j,k in zip(hand[0][:amount], hand[1][:amount],hand[2][:amount])],color=colors[3],label=action_labels[3],width=barWidth)
        elif len(action_labels) == 3:
            axs[i].bar(epochs,hand[0][:amount],color=colors[0],label=action_labels[0],width=barWidth)
            axs[i].bar(epochs,hand[1][:amount],bottom=hand[0][:amount],color=colors[1],label=action_labels[1], width=barWidth)
            axs[i].bar(epochs,hand[2][:amount],bottom=[i+j for i,j in zip(hand[0][:amount], hand[1][:amount])],color=colors[2],label=action_labels[2],width=barWidth)
        elif len(action_labels) == 2:
            axs[i].bar(epochs,hand[0][:amount],color=colors[0],label=action_labels[0],width=barWidth)
            axs[i].bar(epochs,hand[1][:amount],bottom=hand[0][:amount],color=colors[1],label=action_labels[1], width=barWidth)
        else:
            raise ValueError(f'{len(action_labels)} Number of actions not supported')
        axs[i].grid(True)
        axs[i].set_title(f'Hand {hand_labels[i]}')
        # axs[i].set_xlabel('Epochs')
        # axs[i].set_ylabel('Frequency')
        axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #   fancybox=True, shadow=True, ncol=5)
    # axs.legend()
    fig.subplots_adjust(hspace=1)
    fig.savefig(f'{path+title}.png',bbox_inches='tight')
    # plt.title(title)
    # plt.xlabel('Epochs')
    # plt.ylabel('Frequency')
    # plt.legend()
    plt.close()


def plot_betsize_probabilities(training_round=0):
    query = {
        'training_round':training_round
    }
    projection ={'betsizes':1,'hand':1,'_id':0}
    params = {
        'interval':100   
    }
    mongo = MongoDB()
    # SB
    for position in pdt.Positions.ALL:
        query['position'] = position
        data = mongo.get_data(query,projection)
        betsize,unique_hands,unique_betsize = mongo.betsizeByHand(data,params)
        hand_labels = [f'Hand {pdt.Globals.KUHN_CARD_DICT[hand]}' for hand in unique_hands]
        action_labels = [size for size in unique_betsize]
        plot_frequencies(f'{gametype}_betsize_probabilities_for_{query["position"]}',betsize,hand_labels,action_labels)

def plot_action_frequencies_by_hand(actiontype,handtype,training_round=0):
    print(actiontype,handtype)
    query = {
        # 'training_round':training_round
    }
    projection ={'action':1,'hand_strength':1,'hand':1,'_id':0}
    data_params = {
        'interval':10
    }
    mongo = MongoDB()
    gametype = mongo.get_gametype(training_round)
    for position in pdt.Positions.ALL:
        query['position'] = position
        data = mongo.get_data(query,projection)
        if handtype == pdt.VisualHandTypes.HAND:
            actions,hands,unique_actions = mongo.actionByHand(data,data_params)
        else:
            actions,hands,unique_actions = mongo.actionByHandStrength(data,data_params)
        hand_labels = HAND_LABELS_DICT[actiontype](hands)
        action_labels = [ACTION_DICT[act] for act in unique_actions]
        plot_frequencies(f'{gametype}_action_{handtype}_for_{query["position"]}',actions,hand_labels,action_labels)

def plot_action_frequencies(actiontype,training_round=0):
    print(actiontype)
    query = {
        'training_round':training_round
    }
    projection ={'action_probs':1,'_id':0}
    data_params = {
        'interval':1000
    }
    mongo = MongoDB()
    gametype = mongo.get_gametype(training_round)
    for position in pdt.Positions.ALL:
        query['position'] = position
        data = mongo.get_data(query,projection)
        action_probs = []
        for point in data:
            action_probs.append(point['action_probs'])
        action_labels = [ACTION_DICT[act] for act in unique_actions]
        plot_data(f'{gametype}_action_probs_for_{query["position"]}',action_probs,action_labels)

def plot_critic_values(training_round=0):
    query = {
        'position':args.position,
        'training_round':args.run
    }
    projection ={'hand':1,'value':1,'reward':1,'_id':0}
    mongo = MongoDB()
    gametype = mongo.get_gametype(training_round)

    for position in pdt.Positions.ALL:
        query['position'] = position
        data = mongo.get_data(query,projection)
        rewards = []
        values = []
        hands = []
        for point in data:
            rewards.append(point['reward'])
            values.append(point['value'])
            hands.append(point['hand'])
        # plot value loss over time
        interval = 25
        critic_loss = np.array(values) - (np.array(rewards) / 2)
        critic_loss_rolling_mean = []
        for i in range(len(critic_loss)-interval):
            critic_loss_rolling_mean.append(np.mean(critic_loss[i:interval+i]))
        plot_data(f'Critic loss for {query["position"]}',[critic_loss_rolling_mean],['Values'])


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
                        default=pdt.Positions.SB,
                        metavar=f"[{pdt.Positions.SB},{pdt.Positions.BB}]",
                        help='Which position to look at')
    parser.add_argument('--category',
                        default=f'{pdt.VisualCategories.ACTION}',
                        metavar=f"[{pdt.VisualCategories.ACTION},{pdt.VisualCategories.BETSIZE},{pdt.VisualCategories.REWARD}]",
                        help='Chooses the type of data to look at')
    parser.add_argument('--actiontype',
                        default=f'{pdt.VisualActionTypes.FREQUENCY}',
                        metavar=f"[{pdt.VisualActionTypes.FREQUENCY},{pdt.VisualActionTypes.PROBABILITY}]",
                        help='Chooses the type of data to look at')
    parser.add_argument('--handtype',
                        default=f'{pdt.VisualHandTypes.HAND}',
                        metavar=f"[{pdt.VisualHandTypes.HAND},{pdt.VisualHandTypes.HANDSTRENGTH}]",
                        help='Chooses the type of data to look at')

    args = parser.parse_args()
    
    # query = {
    #     'position':args.position,
    #     # 'type':args.type,
    #     # 'step':0,
    #     # 'poker_round':0,
    #     # 'game':args.game,
    #     'training_round':args.run
    # }

    # # projection ={'hand':1,'value':1,'reward':1,'_id':0}
    # projection ={'action':1,'_id':0}
    # params = {
    #     'interval':100
    # }
    # # projection = None
    # mongo = MongoDB()
    # data = mongo.get_data(query,projection)
    # actions = mongo.byActions(data,action_only=True)
    # rewards = mongo.byRewards(query)
    # actions,hands = mongo.actionByHand(query)
    # plot_data(f'Rewards for {query["position"]}',[rewards],['Rewards'])
    assert(args.category in pdt.VisualCategories.ALL)
    assert(args.actiontype in pdt.VisualActionTypes.ALL)
    assert(args.handtype in pdt.VisualHandTypes.ALL)
    if args.category == 'action':
        # plot_action_frequencies(args.actiontype,args.run)
        plot_action_frequencies_by_hand(args.actiontype,args.handtype,args.run)
    elif args.category == 'betsize':
        plot_betsize_probabilities(args.run)
    elif args.category == 'reward':
        plot_critic_values(args.run)
    else:
        raise ValueError('Category not supported')