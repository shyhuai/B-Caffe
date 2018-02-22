from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import os 
import sys
import argparse
import math

class Bar:
    def __init__(self, start_time, duration, max_time, ax, type='p'):
        """
        type: p for compute, m for communication
        """
        self.start_time_ = start_time
        self.ax_ = ax
        self.max_time_ = max_time
        self.duration_ = duration/max_time
        self.type_ = type
        self.y_ = 0.4 if self.type_ is 'p' else 0.3
        self.color_ = 'b' if self.type_ is 'p' else 'g'
        self.height_ = 0.1

    def render(self):
        x = self.start_time_ / self.max_time_
        y = self.y_
        rect =  Rectangle((x, y), self.duration_, self.height_, axes=self.ax_, color=self.color_, ec='black')
        self.ax_.add_patch(rect)
        return rect

def render_log(filename):
    f = open(filename, 'r')
    sizes = []
    computes = []
    comms = []
    for l in f.readlines():
        items = l.split('[')[1][0:-2].split(',')
        items = [float(it.strip()) for it in items]
        if items[1] < 2048 or int(items[2]) == 0:
            continue
        sizes.append(items[1])
        computes.append(items[2])
        #comms.append(items[3])
        comms.append(float(items[4]))
    f.close()
    #sizes = sizes[::-1]
    #computes = computes[::-1]
    #comms = comms[::-1]
    start_time = 0.0
    comm_start_time = 0.0
    comm = 0.0
    max_time = max(np.sum(computes), np.sum(comms)+computes[0])
    fig, ax = plt.subplots(1)
    print('sizes: ', sizes)
    print('computes: ', computes)
    print('communications: ', comms)
    for i in range(len(computes)):
        comp = computes[i]
        bar = Bar(start_time, comp, max_time, ax, type='p')
        bar.render()
        if comm_start_time + comm > start_time + comp:
            comm_start_time = comm_start_time + comm
        else:
            comm_start_time = start_time + comp
        comm = comms[i]
        bar_m = Bar(comm_start_time, comm, max_time, ax, type='m')
        bar_m.render()
        start_time += comp 
    plt.show()
    plt.clf()
    #plt.scatter(sizes, comms)
    #plt.show()

def allreduce_log(filename):
    f = open(filename, 'r')
    num_of_nodes = 2 
    sizes = []
    comms = []
    for l in f.readlines():
        if l[0] == '#' or len(l)<10:
            continue
        items = ' '.join(l.split()).split()
        comm = float(items[-1])
        size = int(items[0].split(',')[1])
        num_of_nodes = int(items[0].split(',')[0])
        comms.append(comm)
        sizes.append(size)
    f.close()
    print('num_of_nodes: ', num_of_nodes)
    print('sizes: ', sizes)
    print('comms: ', comms)
    return num_of_nodes, sizes, comms


def plot_allreduce_log(filenames):
    markers=['-ro', '-go', '-bo']
    for index, fn in enumerate(filenames):
        num_of_nodes, sizes, comms = allreduce_log(fn)
        line1, = plt.plot(sizes, comms, markers[index])
    plt.show()
    plt.clf()


if __name__ == '__main__':
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/alexnet/tmpcomm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/googlenet/tmp2comm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/googlenet/tmp4comm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/googlenet/tmp8comm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/googlenet/tmp8ocomm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/vgg/tmp8comm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/vgg/tmp4comm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/vgg/tmp2comm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/resnet/tmp2comm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/resnet/tmp4comm.log'
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/resnet/tmp8comm.log'
    test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/resnet/tmp8ocomm.log'
    render_log(test_file)

    #test_file = '../logdata/allreduce2.log'
    #allreduce_log(test_file)
    #plot_allreduce_log(['../logdata/allreduce2.log', '../logdata/allreduce4.log','../logdata/allreduce8.log'])
