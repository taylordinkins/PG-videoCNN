import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import csv

import argparse




def make_graph_shown(folder_grow,folder_normal):
    with open(folder_grow + '/images_show_log.txt', 'r') as f:
        reader = csv.reader(f)
        grown = list(reader)

    with open(folder_normal + '/images_show_log.txt', 'r') as f:
        reader = csv.reader(f)
        normal = list(reader)


    X_plot = []
    Y_grown = []
    Y_normal = []

    for i, (x, y) in enumerate(grown):
        X_plot.append(int(y))
        Y_grown.append(float(x))

    for i, (x, y) in enumerate(normal):
        Y_normal.append(float(x))

    #X_plot.sort()

    max_len = min(len(Y_normal), len(Y_grown))

    X_plot = X_plot[:max_len]
    Y_grown = Y_grown[:max_len]
    Y_normal = Y_normal[:max_len]


    print("saving graphs")
    plt.grid(True)
    plt.plot(X_plot, Y_grown)
    plt.plot(X_plot, Y_normal)
    plt.ylabel('time (min)',fontweight='bold')
    plt.xlabel('images shown',fontweight='bold')
    plt.legend(['grown', 'normal'], loc='upper left')
    plt.title('images shown comparison', fontsize=20)
    #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%g')) 
    #plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%g')) 
    plt.savefig(folder_grow+'/images_show_log.png')
    plt.clf()

def make_graph_loss(folder_grow,folder_normal):
    with open(folder_grow + '/loss_log.txt', 'r') as f:
        reader = csv.reader(f)
        grown = list(reader)

    with open(folder_normal + '/loss_log.txt', 'r') as f:
        reader = csv.reader(f)
        normal = list(reader)


    X_plot = []
    Y_grown = []
    Y_normal = []

    for i, (x, y) in enumerate(grown):
        X_plot.append(float(x))
        Y_grown.append(float(y))

    for i, (x, y) in enumerate(normal):
        Y_normal.append(float(y))

    #X_plot.sort()


    
    max_len = max(len(Y_normal), len(Y_grown))
    Y_normal = Y_normal + [Y_normal[-1]] * (max_len - len(Y_normal))

    X_plot = X_plot[10:]
    Y_grown = Y_grown[10:]
    Y_normal = Y_normal[10:]

    print("saving graphs")
    plt.grid(True)
    plt.plot(X_plot, Y_grown)
    plt.plot(X_plot, Y_normal)
    plt.ylabel('loss',fontweight='bold')
    plt.xlabel('time (min)',fontweight='bold')
    plt.legend(['grown', 'normal'], loc='upper left')
    plt.title('loss comparison', fontsize=20)
    #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%g')) 
    #plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%g')) 
    plt.savefig(folder_grow+'/loss_log.png')
    plt.clf()



parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="mmnist_grow1")
args = parser.parse_args()

make_graph_shown('mmnist_grow1','mmnist_grow0')
make_graph_shown('kth_grow1','kth_grow0')

make_graph_loss('mmnist_grow1','mmnist_grow0')
make_graph_loss('kth_grow1','kth_grow0')


#make_graph('mmnist_grow1/', 'images shown', 'images shown grown', 'images_show_log')
#make_graph('kth_grow1/', 'images shown', 'images shown grown', 'images_show_log')




'''
def make_graph(folder, Y_label, title, file):
    with open(folder + file + '.txt', 'r') as f:
        reader = csv.reader(f)
        my_list = list(reader)

    X = []
    Y = []

    for i, (x, y) in enumerate(my_list):
        if i % 50 ==0:
            X.append(x)
            Y.append(y)


    import pdb; pdb.set_trace()

    print("saving graphs")
    plt.grid(True)
    plt.plot(X, Y)
    plt.ylabel(Y_label,fontweight='bold')
    plt.xlabel('time (min)',fontweight='bold')
    #plt.legend(['grown', 'normal'], loc='upper left')
    plt.title(title, fontsize=20)
    #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%g')) 
    #plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%g')) 
    plt.savefig(folder + file + '.png')
    plt.clf()
    '''

