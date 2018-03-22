from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from utils import read_log, plot_hist, update_fontsize, autolabel, read_p100_log
from plot_sth import Bar
import plot_sth as Color

OUTPUT_PATH = '/media/sf_Shared_Data/tmp/sc18'

num_of_nodes = [2, 4, 8, 16]
num_of_nodes = [2, 4, 8]
#num_of_nodes = [8, 80, 81, 82, 83, 85]
#num_of_nodes = [16, 32, 64]
B = 9.0 * 1024 * 1024 * 1024.0 / 8 # 10 Gbps Ethernet
#B = 56 * 1024 * 1024 * 1024.0 / 8 # 56 Gbps IB
markers = {2:'o',
        4:'x',
        8:'^'}

def time_of_allreduce(n, M, B=B):
    """
    n: number of nodes
    M: size of message
    B: bandwidth of link
    """
    # Model 1, TernGrad, NIPS2017
    #if True:
    #    ncost = 100 * 1e-6
    #    nwd = B
    #    return ncost * np.log2(n) + M / nwd * np.log2(n) 

    # Model 2, Lower bound, E. Chan, et al., 2007
    if True:
        #alpha = 7.2*1e-6 #Yang 2017, SC17, Scaling Deep Learning on GPU and Knights Landing clusters
        #alpha = 6.25*1e-6*n # From the data gpuhome benchmark
        #alpha = 12*1e-6*n # From the data gpuhome benchmark
        alpha = 45.25*1e-6#*np.log2(n) # From the data gpuhome benchmark
        beta =  1 / B *1.2
        gamma = 1.0 / (16.0 * 1e9  * 4) * 160
        M = 4*M
        t = 2*(n)*alpha + 2*(n-1)*M*beta/n + (n-1)*M*gamma/n
        return t * 1e6
    ts = 7.5/ (1000.0 * 1000)# startup time in second
    #seconds = (np.ceil(np.log2(n)) + n - 1) * ts + (2*n - 1 + n-1) * M / n * 1/B 
    #seconds = (np.ceil(np.log2(n)) + n - 1) * ts + 2 * (n - 1) * 2*M/n * 1/B
    #tcompute = 1. / (2.2 * 1000 * 1000 * 1000)
    tcompute = 1. / (1 * 1000 * 1000 * 1000)
    #seconds = 2 * (n - 1) * ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute
    #C = 1024.0 * 1024 # segmented_size 
    #if M > C * n:
    #    # ring_segmented allreduce
    #    seconds = (M / C + (n - 2)) * (ts + C / B + C * tcompute)
    #else:
        # ring allreduce, better than the above
        #seconds = (n - 1) * ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute
    seconds = 2*(n-1)*n*ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute

    #C =  512.0
    #seconds = (M / C + n-2) * (ts + C/B)
    return seconds * 1000 * 1000 # micro seconds



class Simulator():
    def __init__(self, name, computes, sizes, num_of_nodes, render=True):
        self.name = name
        self.computes = computes
        self.sizes = sizes
        self.num_of_nodes = num_of_nodes
        self.comms = None
        self.title = name + ' (WFBP)'
        self.max_time = 0
        self.ax = None
        self.render = render
        self.merged_layers = []

    def wfbp(self, with_optimal=False):
        start_time = 0.0
        comm_start_time = 0.0
        comm = 0.0
        if not self.comms:
            comms = [time_of_allreduce(self.num_of_nodes, s, B) for s in self.sizes]
        else:
            comms = self.comms
        max_time = max(np.sum(self.computes), np.sum(comms)+self.computes[0])
        if not with_optimal:
            self.max_time = max_time
        if not self.ax and self.render:
            fig, ax = plt.subplots(1, figsize=(30, 3))
            #ax.set_title(self.title, x=0.5, y=0.8)
            self.ax = ax
        comm_layer_id = ''
        for i in range(len(self.computes)):
            comp = self.computes[i]
            layer_id = len(self.computes) - i
            if not with_optimal:
                if self.render:
                    bar = Bar(start_time, comp, self.max_time, self.ax, type='p', index=layer_id)
                    bar.render()
            if comm_start_time + comm > start_time + comp:
                comm_start_time = comm_start_time + comm
            else:
                comm_start_time = start_time + comp
            if comm == 0.0 and comm_layer_id != '':
                comm_layer_id = str(comm_layer_id)+','+str((len(self.computes) - i))
            else:
                comm_layer_id = str(layer_id)

            comm = comms[i]
            type = 'wc'
            if with_optimal:
                type = 'mc'
            if self.render:
                bar_m = Bar(comm_start_time, comm, self.max_time, self.ax, type=type, index=comm_layer_id, is_optimal=with_optimal)
                bar_m.render()
            start_time += comp 
        total_time = (comm_start_time + comm)/1000.0
        print('Total time: ', total_time, ' ms')
        if self.render:
            plt.subplots_adjust(left=0.06, right=1.)
        return total_time

    def synceasgd(self):
        start_time = 0.0
        comm_start_time = 0.0
        comm = 0.0
        total_size = np.sum(self.sizes)
        comm = time_of_allreduce(self.num_of_nodes, total_size, B)
        total_comp = np.sum(self.computes)
        comm_start_time = total_comp
        index = ','.join([str(len(self.computes)-i) for i in range(0, len(self.computes))])
        if self.render:
            bar = Bar(np.sum(self.computes), comm, self.max_time, self.ax, type='sc', index=index)
            bar.render()
        total_time = (comm_start_time + comm)/1000.0
        print('Total time: ', total_time, ' ms')
        if self.render:
            pass
        return total_time



    def cal_comm_starts(self, comms, comps):
        """
        comms and comps have been aligned
        """
        start_comms = []
        start_comms.append(0.0)
        sum_comp = 0.0
        for i in range(1, len(comms)):
            comm = comms[i-1]
            comp = comps[i-1]
            #print(start_comms[i-1],comm, sum_comp,comp)
            start_comm = max(start_comms[i-1]+comm, sum_comp+comp)
            #print('start_comm: ', start_comm, ', comm: ', comm)
            start_comms.append(start_comm)
            sum_comp += comp
        return start_comms

    def merge(self, comms, sizes, i, p, merge_size, comps):
        comms[i] = 0# merge here
        comms[i+1] = p
        sizes[i+1] = merge_size 
        start_comms = self.cal_comm_starts(comms, comps)
        #print('start_comms: ', start_comms)
        self.merged_layers.append(i)
        return start_comms

    def gmwfbp2(self):
        if not self.comms:
            comms = [time_of_allreduce(self.num_of_nodes, s, B) for s in self.sizes]
        else:
            comms = self.comms

        #comms = comms[0:-1]
        #print('comms: ', comms)
        comps = self.computes[1:]
        comps.append(0) # for last communication

        optimal_comms = list(comms)
        optimal_sizes = list(self.sizes)
        start_comms = self.cal_comm_starts(optimal_comms, comps)
        sum_comp = 0.0
        #print('start_comms: ', start_comms)
        #return

        for i in range(0, len(comms)-1):
            comp = comps[i]
            comm = optimal_comms[i]
            if start_comms[i] + comm > comp+sum_comp:
                # cannot be hidden, so we need to merge
                merge_size = optimal_sizes[i+1] + optimal_sizes[i]
                r = comm + optimal_comms[i+1]
                p = time_of_allreduce(self.num_of_nodes, merge_size, B) 
                if start_comms[i] >= comp+sum_comp:
                    # don't care about computation
                    if p < r:
                        start_comms = self.merge(optimal_comms, optimal_sizes, i, p, merge_size, comps)
                        #optimal_comms[i] = 0# merge here
                        #optimal_comms[i+1] = p
                        #optimal_sizes[i+1] += merge_size 
                        #start_comms = self.cal_comm_starts(optimal_comms, comps)
                else:
                    if comp+sum_comp+p < start_comms[i]+comm+optimal_comms[i+1]:
                        start_comms = self.merge(optimal_comms, optimal_sizes, i, p, merge_size, comps)
            else:
                pass # optimal, nothing to do
            sum_comp += comp
        optimal_comms.append(comms[-1])
        self.wfbp()
        self.synceasgd()
        self.comms = optimal_comms
        self.title = self.name+ ' (GM-WFBP)'
        ret = self.wfbp(with_optimal=True)
        #print('merged-layers: ', self.merged_layers)
        return ret


def read_allreduce_log(filename):
    f = open(filename, 'r')
    sizes = []
    comms = []
    for l in f.readlines():
        if l[0] == '#' or len(l)<10 :
            continue
        items = ' '.join(l.split()).split()
        comm = float(items[-1])
        #size = int(items[0].split(',')[1])
        size = int(items[0])/4
        #if size > 2e3:
        #    break
        if size < 2048 or size > 2e4:
            continue
        #num_of_nodes = int(items[0].split(',')[0])
        comms.append(comm)
        sizes.append(size)
    f.close()
    #print('num_of_nodes: ', num_of_nodes)
    #print('sizes: ', sizes)
    #print('comms: ', comms)
    return sizes, comms, []


def predict(filename, n, color, marker, label, sizes=None, ax=None):
    #sizes, comms, comps, merged_comms = read_log(filename)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4.5))
    if sizes is None:
        sizes, comms, comps = read_allreduce_log(filename)
    #plt.scatter(range(1, len(sizes)+1), sizes, c=color, label=label, marker=marker, s=40, facecolors='none', edgecolors=color)
        plt.plot(sizes, comms, c=color, marker=marker, label=label+' measured', linewidth=2)
        #plt.plot(sizes, comms, c=color, marker=marker, label=label, linewidth=2)
    #bandwidths = np.array(sizes)/np.array(comms)
    #plt.plot(sizes, bandwidths, c=color, marker=marker, label=label, linewidth=2)
    predicts = []
    for M in sizes:
       p = time_of_allreduce(n, M, B) 
       predicts.append(p)
    #rerror = (np.array(predicts)-np.array(comms))/np.array(comms)
    #print('erro: ', np.mean(np.abs(rerror)))
    #plt.scatter(sizes, predicts, c='red', marker=markers[n])
    ax.plot(sizes, predicts, c=color, marker=marker, linestyle='--', label=label+' predict', markerfacecolor='white', linewidth=1)
    return sizes

def plot_all_communication_overheads():
    #labels = ['2-node', '4-node', '8-node', '16-node']
    fig, ax = plt.subplots(figsize=(5,4.5))
    labels = ['%d-node' % i for i in num_of_nodes]
    colors = ['r', 'g', 'b', 'black', 'y', 'c']
    markers = ['^', 'o', 'd', '*', 'x', 'v']
    sizes = None
    sizes = np.arange(128.0, 1e5, step=8192)
    for i, n in enumerate(num_of_nodes):
        test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/allreduce%d.log' % n 
        #test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/t716/allreduce%d.log' % n  # 1Gbps
        #test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/t716/ompi2.1log/allreduce%d.log' % n  # 1Gbps
        #test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/t716/ompi3.0log/allreduce%d.log' % n  # 1Gbps
        #sizes = predict(test_file, n, colors[i], markers[i], labels[i])
        predict(test_file, n, colors[i], markers[i], labels[i], sizes, ax)
    #plt.xlim(left=0)
    #plt.xlabel('Message size (bytes)')
    #ax.ticklabel_format(style='sci',axis='x')
    plt.xlabel('# of parameters')
    plt.ylabel(r'Latency ($\mu$s)')
    plt.ylim(bottom=0, top=plt.ylim()[1]+200)
    #plt.xscale("log", nonposy='clip')
    plt.legend(ncol=1, loc=2)
    update_fontsize(ax, fontsize=14)
    plt.subplots_adjust(left=0.18, bottom=0.13, top=0.91, right=0.92)
    #plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'commtime'))
    plt.show()

def gmwfbp_simulate():
    name = 'GoogleNet'
    #name = 'ResNet'
    #name = 'VGG'
    #name = 'DenseNet'
    num_of_nodes = 32
    test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8comm.log' % name.lower()
    sizes, comms, computes, merged_comms = read_log(test_file)
    #computes = [c/4 for c in computes]
    #sizes = [1., 1., 1., 1.]
    #computes = [3., 3.5, 5., 6.]
    #sim = Simulator(name, computes[0:4], sizes[0:4], num_of_nodes)
    sim = Simulator(name, computes, sizes, num_of_nodes)
    #sim.wfbp()
    sim.gmwfbp2()
    plt.savefig('%s/breakdown%s.pdf' % (OUTPUT_PATH, name.lower()))
    #plt.show()

def gmwfbp_speedup():
    #configs = ['GoogleNet', 64]
    configs = ['ResNet', 32]
    #configs = ['DenseNet', 128]
    name = configs[0] 
    b = configs[1]
    test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8comm.log' % name.lower()
    sizes, comms, computes, merged_comms = read_log(test_file)
    device = 'k80'

    device = 'p100'
    pfn = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8commp100%s.log' % (name.lower(), name.lower())
    val_sizes, computes = read_p100_log(pfn)
    print('computes: ', np.sum(computes))
    print('computes: ', computes)
    assert len(computes) == len(sizes)

    nnodes = [4, 8, 16, 32, 64]
    #nnodes = [2, 4, 8]
    wfbps = []
    gmwfbps = []
    synceasgds = []
    micomputes = np.array(computes)
    tf = np.sum(micomputes) * 0.5 / 1000
    tb = np.sum(micomputes) / 1000
    total_size = np.sum(sizes)
    single = b/(tf+tb)
    optimal = []
    colors = ['k', 'r', 'g', 'b']
    markers = ['s', '^', 'o', 'd']
    for num_of_nodes in nnodes:
        sim = Simulator(name, computes, sizes, num_of_nodes, render=False)
        wfbp = sim.wfbp()
        wfbps.append(b*num_of_nodes/(wfbp+tf)/single)
        gmwfbp = sim.gmwfbp2()
        gmwfbps.append(b*num_of_nodes/(gmwfbp+tf)/single)
        tc = time_of_allreduce(num_of_nodes, total_size, B)/1000
        print('#nodes:', num_of_nodes, ', tc: ', tc) 
        synceasgd = tb + tf + tc
        synceasgds.append(b*num_of_nodes/synceasgd/single)
        optimal.append(num_of_nodes)
    print('tf: ', tf)
    print('tb: ', tb) 
    print('total_size: ', total_size)
    print('wfbp: ', wfbps)
    print('gmwfbps: ', gmwfbps)
    print('synceasgds: ', synceasgds)
    print('compared to synceasgds: ', np.array(gmwfbps)/np.array(synceasgds))
    print('compared to wfbps: ', np.array(gmwfbps)/np.array(wfbps))
    fig, ax = plt.subplots(figsize=(5,4.5))
    ax.plot(nnodes, optimal, color='k', marker='s', label='Linear')
    ax.plot(nnodes, wfbps, color='r', marker='d', label='WFBP')
    ax.plot(nnodes, synceasgds, color='b', marker='o', label='SyncEASGD')
    ax.plot(nnodes, gmwfbps, color='g', marker='^', label='MG-WFBP')
    plt.legend(loc=2)
    plt.xlabel('# of nodes')
    plt.ylabel('Speedup')
    #plt.title('%s-Simulation'%name)
    #plt.yscale('log', basey=2)
    #plt.xscale('log', basey=2)
    plt.ylim(bottom=1,top=nnodes[-1]+1)
    plt.xlim(left=1, right=nnodes[-1]+1)
    plt.xticks(nnodes)
    plt.yticks(nnodes)
    plt.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    update_fontsize(ax, fontsize=14)
    plt.subplots_adjust(left=0.13, bottom=0.13, top=0.96, right=0.97)
    plt.savefig('%s/speedup%s.pdf' % (OUTPUT_PATH, name.lower()+device))
    #plt.show()

def plot_realdata_comm(datas, configs):
    def calculate_real_comms(data, bs):
        times = [bs/((d/2)/2**(i-1)) for i, d in enumerate(data)]
        comp = times[0]
        comms = [t-times[0] for t in times[1:]]
        return comp, comms
    fig, ax = plt.subplots(figsize=(4.8,3.4))
    count = len(datas[0][1:])
    ind = np.arange(count)
    width = 0.25
    s = -int(count/2)
    print('s: ', s)
    margin = 0.05
    xticklabels = [str(2**(i+1)) for i in range(count)]
    s = (1 - (width*count+(count-1) *margin))/2+width
    ind = np.array([s+i+1 for i in range(count)])
    centerind = None
    labels=['WF.', 'S.E.', 'M.W.']
    for i, data in enumerate(datas):
        comp, comms= calculate_real_comms(data, configs[1])
        comps = [comp for j in comms]
        newind = ind+s*width+(s+1)*margin
        p1 = ax.bar(newind, comps, width, color=Color.comp_color,hatch='x', label='Comp.')
        p2 = ax.bar(newind, comms, width,
                             bottom=comps, color=Color.comm_color, label='Comm.')

        s += 1 
        autolabel(p2, ax, labels[i], 0)
        print('comp: ', comp)
        print('comms: ', comms)
        print('')

    rects = ax.patches
    ax.text(10, 10, 'ehhlo', color='b')
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend([handles[0][0]], [labels[0][0]], ncol=2)
    print(labels)
    print(handles)
    ax.set_xlim(left=1+0.3)
    ax.set_ylim(top=ax.get_ylim()[1]*1.3)
    ax.set_xticks(ind+2*(width+margin))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('# of nodes')
    ax.set_ylabel('Time [s]')
    update_fontsize(ax, 14)
    ax.legend((p1[0], p2[0]), (labels[0],labels[1] ), ncol=2, handletextpad=0.2, columnspacing =1.)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.17, top=0.94)
    plt.savefig('%s/comm%sreal.pdf' % (OUTPUT_PATH, configs[0].lower()))
    #plt.show()



def realdata_speedup():
    configs = ['GoogleNet', 64]
    wfbps =     [81.68*2, 74.83*2*2, 74.91*2*4, 2*62.9*8]
    gmwfbps =   [81.68*2, 79.02*2*2, 75.03*2*4, 2*75.68*8]
    synceasgds =[81.68*2, 62.57*2*2, 57.67*2*4, 2*55.58*8]
    device = 'k80'
    configs = ['ResNet', 32]
    wfbps =     [76.85, 75.55*2, 73.679*4, 58.2*8]
    gmwfbps =   [76.85, 75.59*2, 73.8*4, 70.8251*8]
    synceasgds =[76.85, 60.0*2, 55.7*4, 50.8*8]
    datas = [wfbps, synceasgds, gmwfbps]
    #plot_realdata_comm(datas, configs)
    #return

    #configs = ['DenseNet', 128]
    name = configs[0] 
    b = configs[1]
    nnodes = [2, 4, 8]

    fig, ax = plt.subplots(figsize=(5,4.5))
    optimal = nnodes 
    wfbps = [i/wfbps[0] for i in wfbps[1:]]
    gmwfbps = [i/gmwfbps[0] for i in gmwfbps[1:]]
    synceasgds= [i/synceasgds[0] for i in synceasgds[1:]]
    print('compared to wfbp: ', np.array(gmwfbps)/np.array(wfbps))
    print('compared to synceasgds: ', np.array(gmwfbps)/np.array(synceasgds))
    ax.plot(nnodes, optimal, color='k', marker='s', label='Linear')
    ax.plot(nnodes, wfbps, color='r', marker='d', label='WFBP')
    ax.plot(nnodes, synceasgds, color='b', marker='o', label='SyncEASGD')
    ax.plot(nnodes, gmwfbps, color='g', marker='^', label='MG-WFBP')
    #plt.yscale('log', basey=2)
    #plt.xscale('log', basey=2)
    plt.legend(loc=2)
    plt.xlabel('# of nodes')
    plt.ylabel('Speedup')
    plt.xticks(nnodes)
    plt.yticks(nnodes)
    plt.ylim(bottom=1,top=nnodes[-1]+1)
    plt.xlim(left=1, right=nnodes[-1]+1)
    plt.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    #plt.title('%s-Realworld'%name)
    update_fontsize(ax, fontsize=14)
    plt.subplots_adjust(left=0.13, bottom=0.13, top=0.96, right=0.97)
    plt.savefig('%s/speedup%sreal.pdf' % (OUTPUT_PATH, name.lower()+device))
    #plt.show()

def parse_real_comm_cost():
    configs = ['GoogleNet', 'gm'] #SyncEASGD
    name = configs[0]
    t = configs[1] 
    nnodes = [2, 4, 8]
    ncomms = []
    for n in nnodes:
        test_file = '/home/shshi/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/%s%dcomm.log' % (name.lower(), t, n)
        sizes, comms, computes, merged_comms = read_log(test_file)
        ncomms.append(np.sum(merged_comms))
    print('network: ', name, ', type: ', t)
    print('ncomms: ', ncomms)


def speedup_with_r_and_n(r, n):
    return n/(1.+r)

def draw_ssgd_speedup():
    Ns = [8, 16, 32, 64]
    r = np.arange(0, 4, step=0.1)
    for N in Ns:
        s = N / (1+r)
        plt.plot(r, s)
    #plt.yscale('log', basey=2)
    plt.show()


if __name__ == '__main__':
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/googlenet/tmp8ocomm.log'
    #read_log(test_file)
    #test_file = '../logdata/allreduce%d.log' % num_of_nodes 
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/allgather%d.log' % num_of_nodes 
    #plot_all_communication_overheads()
    #gmwfbp_simulate()
    #realdata_speedup()
    #parse_real_comm_cost()
    gmwfbp_speedup()
    #draw_ssgd_speedup()

