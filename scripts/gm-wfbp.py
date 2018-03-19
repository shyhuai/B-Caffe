from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from utils import read_log, plot_hist
from plot_sth import Bar

OUTPUT_PATH = '/media/sf_Shared_Data/tmp/sc18'

num_of_nodes = [2, 4, 8, 16]
#num_of_nodes = [2, 4, 8]
#num_of_nodes = [8, 80, 81, 82, 83, 85]
num_of_nodes = [16, 32, 64]
B = 9.37 * 1024 * 1024 * 1024.0 / 8 # 10 Gbps Ethernet
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
        #alpha = 50.0*1e-6
        alpha = 7.2*1e-6 #Yang 2017, SC17, Scaling Deep Learning on GPU and Knights Landing clusters
        beta =  1 / B
        gamma = 1.0 / (16.0 * 1e9  * 4 / 2)
        M = 4*M
        t = 2*n*alpha + 2*(n-1)*M*beta/n + (n-1)*M*gamma/n
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



    def _search_unmerged(self, start_idx, end_idx, computes, comms, flags):
        """
        start_idx should not less than 1
        flags all false values, means all are bad cases
        """
        if start_idx == end_idx:
            return -1  # End, no need to merge 
        i = start_idx
        start_comp_time = computes[i-1] 
        start_comm_time = computes[i-1]
        optimal_found = False
        while i < end_idx:
            comm = comms[i-1]
            comp = computes[i]
            print('comm=%f, comp=%f'%(comm, comp))
            if start_comm_time + comm > start_comp_time + comp:
                start_comm_time += comm# bad case, need to merge
            else:
                # optimal case
                for j in range(start_idx-1, i):
                    flags[j] = True # found
                optimal_found = True
                print('optimal found at ', i)
                return i+1
            start_comp_time += comp 
            i += 1
        return start_idx # this is bad case

    def _merge_gradients(self, start_idx, end_idx, computes, comms, flags, comm_start=0.0):
        if start_idx >= end_idx:
            return start_idx 
        i = start_idx
        start_comp_time = computes[i]
        start_comm_time = comms[i-1] + comm_start
        assert start_comm_time > start_comp_time, 'Error!!, start_comm_time should be larger than start_comp_time'
        sum_size = self.sizes[i-1]
        i += 1
        while i < end_idx:
            comm = comms[i-1]
            comp = computes[i]
            size = self.sizes[i-1]
            sum_size += size
            ptime = time_of_allreduce(self.num_of_nodes, sum_size, B)
            print('[%d]ptime: %f, start_comp_time: %f, start_comm_time: %f, comm: %f, sum_size:%f'% (i, ptime, start_comp_time, start_comm_time, comm, sum_size))

            if ptime + start_comp_time < start_comm_time + comm:
                # Merge
                comms[i-2] = 0.0
                comms[i-1] = ptime
                print('Merged i: ', i-2)
                
                #self.sizes[i-1] += self.sizes[i-2] 
                #self.sizes[i-2] = 0.0
                if ptime <= comp:
                    print('Case 3: optimal')
                    for j in range(start_idx-1, i):
                        flags[i] = True
                    return i+1
                else:
                    print('Case 4: partially optimal')
                    new_comm_start = max(ptime+start_comm_time-(start_comp_time+comp) , 0)
                    i = self._merge_gradients(i+1, end_idx, computes, comms, flags, new_comm_start)
                    #i = self._merge_gradients(i, end_idx, computes, comms, flags)
            else:
                i = self._merge_gradients(i+1, end_idx, computes, comms, flags, comm_start=start_comm_time-start_comp_time)
        return i

    def gmwfbp(self):
        # Merge gradients
        num_of_layers = len(self.computes)
        flags = [False for i in self.computes]
        comms = [time_of_allreduce(self.num_of_nodes, s, B) for s in self.sizes]
        #comms = [100.0 for i in self.computes] 
        print('flags: ', flags)
        print('sizes: ', self.sizes)
        print('comms: ', comms)
        print('computes: ', self.computes)
        optimal_comms = list(comms)
        i = 1
        while i < num_of_layers:
            idx = self._search_unmerged(i, num_of_layers, self.computes, optimal_comms, flags)
            if idx == i:
                #Not found
                i = self._merge_gradients(i, num_of_layers, self.computes, optimal_comms, flags)
            else:
                i = idx
        print('idx: ', idx)
        print('flags: ', flags)
        print('optimal_comms: ', optimal_comms)
        self.wfbp()
        self.comms = optimal_comms
        self.title = self.name+ ' (GM-WFBP)'
        ret = self.wfbp(with_optimal=True)
        plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, self.name.lower()+'_n%d'%self.num_of_nodes))
        return ret

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
        return start_comms

    def gmwfbp2(self):
        if not self.comms:
            comms = [time_of_allreduce(self.num_of_nodes, s, B) for s in self.sizes]
        else:
            comms = self.comms

        #comms = comms[0:-1]
        print('comms: ', comms)
        comps = self.computes[1:]
        comps.append(0) # for last communication

        optimal_comms = list(comms)
        optimal_sizes = list(self.sizes)
        start_comms = self.cal_comm_starts(optimal_comms, comps)
        sum_comp = 0.0
        print('start_comms: ', start_comms)
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
        size = int(items[0])
        #if size > 2e3:
        #    break
        if size < 2048 or size > 2e5:
            continue
        #num_of_nodes = int(items[0].split(',')[0])
        comms.append(comm)
        sizes.append(size)
    f.close()
    #print('num_of_nodes: ', num_of_nodes)
    print('sizes: ', sizes)
    print('comms: ', comms)
    return sizes, comms, []


def predict(filename, n, color, marker, label, sizes=None):
    #sizes, comms, comps = read_log(filename)
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
    plt.plot(sizes, predicts, c=color, marker=marker, linestyle='--', label=label+' predict', markerfacecolor='white', linewidth=2)
    return sizes

def plot_all_communication_overheads():
    #labels = ['2-node', '4-node', '8-node', '16-node']
    labels = ['%d-node' % i for i in num_of_nodes]
    colors = ['r', 'g', 'b', 'black', 'y', 'c']
    markers = ['^', 'o', 'd', '*', 'x', 'v']
    sizes = None
    sizes = np.arange(128.0, 2e5, step=8192)
    for i, n in enumerate(num_of_nodes):
        #test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/allreduce%d.log' % n 
        #test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/t716/allreduce%d.log' % n  # 1Gbps
        test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/t716/ompi2.1log/allreduce%d.log' % n  # 1Gbps
        #test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/t716/ompi3.0log/allreduce%d.log' % n  # 1Gbps
        #sizes = predict(test_file, n, colors[i], markers[i], labels[i])
        predict(test_file, n, colors[i], markers[i], labels[i], sizes)
    #plt.xlim(left=0)
    #plt.xlabel('Message size (bytes)')
    plt.xlabel('# of parameters')
    plt.ylabel(r'Latency ($\mu$s)')
    plt.ylim(bottom=0, top=plt.ylim()[1]+200)
    #plt.xscale("log", nonposy='clip')
    plt.legend(ncol=1, loc=2)
    plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'commtime'))
    #plt.show()

def gmwfbp_simulate():
    name = 'GoogleNet'
    #name = 'ResNet'
    #name = 'VGG'
    #name = 'DenseNet'
    num_of_nodes = 32
    test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8comm.log' % name.lower()
    sizes, comms, computes = read_log(test_file)
    computes = [c/4 for c in computes]
    #sizes = [1., 1., 1., 1.]
    #computes = [3., 3.5, 5., 6.]
    #sim = Simulator(name, computes[0:4], sizes[0:4], num_of_nodes)
    sim = Simulator(name, computes, sizes, num_of_nodes)
    #sim.wfbp()
    sim.gmwfbp2()
    plt.savefig('%s/breakdown%s.pdf' % (OUTPUT_PATH, name.lower()))
    #plt.show()

def gmwfbp_speedup():
    #configs = ['GoogleNet', 128]
    configs = ['ResNet', 128]
    #configs = ['DenseNet', 128]
    name = configs[0] 
    b = configs[1]
    test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8comm.log' % name.lower()
    sizes, comms, computes = read_log(test_file)
    device = 'p100'
    #device = 'k80'
    computes = [c/4 for c in computes] # P100
    nnodes = [4, 8, 16, 32, 64, 80]
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
        print('tc: ', tc) 
        synceasgd = tb + tf + tc
        synceasgds.append(b*num_of_nodes/synceasgd/single)
        optimal.append(num_of_nodes)
    print('tf: ', tf)
    print('tb: ', tb) 
    print('total_size: ', total_size)
    print('wfbp: ', wfbps)
    print('gmwfbps: ', gmwfbps)
    print('synceasgds: ', synceasgds)
    plt.plot(nnodes, optimal, color='k', marker='s', label='Linear')
    plt.plot(nnodes, wfbps, color='r', marker='d', label='WFBP')
    plt.plot(nnodes, synceasgds, color='b', marker='o', label='SyncEASGD')
    plt.plot(nnodes, gmwfbps, color='g', marker='^', label='MG-WFBP')
    print(np.array(gmwfbps)/np.array(synceasgds))
    #plt.yscale('log', basey=2)
    #plt.xscale('log', basey=2)
    plt.legend(loc=2)
    plt.xlabel('# of nodes')
    plt.ylabel('Speedup')
    plt.xticks(nnodes)
    plt.yticks(nnodes)
    plt.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    plt.savefig('%s/speedup%s.pdf' % (OUTPUT_PATH, name.lower()+device))
    #plt.show()

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
    gmwfbp_simulate()
    #gmwfbp_speedup()
    #draw_ssgd_speedup()

