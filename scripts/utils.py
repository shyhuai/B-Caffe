import numpy as np
import matplotlib.pyplot as plt

def read_log(filename):
    f = open(filename, 'r')
    sizes = []
    computes = []
    comms = []
    merged_comms = []
    for l in f.readlines():
        items = l.split('[')[1][0:-2].split(',')
        items = [float(it.strip()) for it in items]
        if int(items[2]) == 0:# or int(items[1]) > 1000000:
            continue
        sizes.append(float(items[1])*4)
        computes.append(items[2])
        #comms.append(items[3])
        comms.append(float(items[4]))
        merged_comms.append(items[4])
    f.close()
    print('sizes: ', sizes)
    print('computes: ', computes)
    print('communications: ', comms)
    return sizes, comms, computes


def plot_hist(d):
    d = np.array(d)
    flatten = d.ravel()
    mean = np.mean(flatten)
    std = np.std(flatten)
    count, bins, ignored = plt.hist(flatten, 100, normed=True)
    print 'mean: %.3f, std: %.3f' % (mean, std)
    n_neg = flatten[np.where(flatten<=0.0)].size
    print '# of zero: %d' % n_neg
    print '# of total: %d' % flatten.size 
    #return n_neg, flatten.size # return #negative, total
    plt.ylabel('Propability')
    plt.xlabel('Nudule Size')
    return flatten

