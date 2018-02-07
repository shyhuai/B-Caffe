from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import os 
import sys
import argparse

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
        if items[1] < 4096:
            continue
        sizes.append(items[1])
        computes.append(items[2])
        comms.append(items[3])
    f.close()
    sizes = sizes[::-1]
    computes = computes[::-1]
    comms = comms[::-1]
    start_time = 0.0
    comm_start_time = 0.0
    comm = 0.0
    max_time = max(np.sum(computes), np.sum(comms)+computes[0])
    fig, ax = plt.subplots(1)
    print('computes: ', computes)
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


if __name__ == '__main__':
    #test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/alexnet/tmpcomm.log'
    test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/googlenet/tmpcomm.log'
    render_log(test_file)
