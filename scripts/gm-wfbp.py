from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

num_of_nodes = [2, 4, 8]
B = 9.37 * 1024 * 1024 * 1024.0 / 8
markers = {2:'o',
        4:'x',
        8:'^'}

def time_of_allreduce(n, M, B):
    """
    n: number of nodes
    M: size of message
    B: bandwidth of link
    """
    ts = 15 / (1000.0 * 1000)# startup time in second
    #seconds = (np.ceil(np.log2(n)) + n - 1) * ts + (2*n - 1 + n-1) * M / n * 1/B 
    #seconds = (np.ceil(np.log2(n)) + n - 1) * ts + 2 * (n - 1) * 2*M/n * 1/B
    #tcompute = 1. / (2.2 * 1000 * 1000 * 1000)
    tcompute = 1. / (1 * 1000 * 1000 * 1000)
    seconds = 2 * (n - 1) * ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute
    C = 1024.0 * 1024 # segmented_size 
    if M > C * n:
        # ring_segmented allreduce
        seconds = (M / C + (n - 2)) * (ts + C / B + C * tcompute)
    else:
        # ring allreduce, better than the above
        #seconds = (n - 1) * ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute
        seconds = (n-1)*n*ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute

    #C =  512.0
    #seconds = (M / C + n-2) * (ts + C/B)
    return seconds * 1000 * 1000 # micro seconds


def read_log(filename):
    f = open(filename, 'r')
    sizes = []
    computes = []
    comms = []
    merged_comms = []
    for l in f.readlines():
        items = l.split('[')[1][0:-2].split(',')
        items = [float(it.strip()) for it in items]
        if int(items[2]) == 0 or int(items[1]) < 10000:
            continue
        sizes.append(items[1])
        computes.append(items[2])
        comms.append(items[3])
        #comms.append(float(items[4]))
        merged_comms.append(items[4])
    f.close()
    print('sizes: ', sizes)
    print('computes: ', computes)
    print('communications: ', comms)
    return sizes, comms, computes

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
        if size < 1024:
            continue
        #num_of_nodes = int(items[0].split(',')[0])
        comms.append(comm)
        sizes.append(size)
    f.close()
    #print('num_of_nodes: ', num_of_nodes)
    print('sizes: ', sizes)
    print('comms: ', comms)
    return sizes, comms, []


def predict(filename, n):
    #sizes, comms, comps = read_log(filename)
    sizes, comms, comps = read_allreduce_log(filename)
    #plt.scatter(sizes, comms, c='blue',marker=markers[n])
    plt.plot(sizes, comms, c='blue',marker=markers[n])
    predicts = []
    for M in sizes:
       p = time_of_allreduce(n, M, B) 
       predicts.append(p)
    #plt.scatter(sizes, predicts, c='red', marker=markers[n])
    plt.plot(sizes, predicts, c='red', marker=markers[n])
    #plt.show()


if __name__ == '__main__':
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/googlenet/tmp8ocomm.log'
    #read_log(test_file)
    #test_file = '../logdata/allreduce%d.log' % num_of_nodes 
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/allgather%d.log' % num_of_nodes 
    for n in num_of_nodes:
        test_file = '/media/sf_Shared_Data/gpuhome/repositories/mpibench/allreduce%d.log' % n 
        predict(test_file, n)
    plt.show()
