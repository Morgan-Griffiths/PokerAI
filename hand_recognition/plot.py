import matplotlib.pyplot as plt

COLORS = ['g','b','m','r','y']

def plot_data(title:str,data:list,labels:list,path='assets/'):
    print(path+title)
    epochs = range(1,len(data[0])+1)
    for i,data_group in enumerate(data):
        plt.plot(epochs,data_group,COLORS[i],label=labels[i])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path+title}.png',bbox_inches='tight')
    plt.close()