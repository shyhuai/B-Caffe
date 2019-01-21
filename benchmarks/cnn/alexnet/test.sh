mpirun -np 2 -hostfile cluster2 caffe train -solver=alexnet-b1024-GPU-solver1.prototxt -gpu=0,1,2,3
#mpirun -np 4 -hostfile cluster4 caffe train -solver=alexnet-b1024-GPU-solver1.prototxt -gpu=0,1,2,3
#mpirun -np 8 -hostfile cluster8 caffe train -solver=alexnet-b1024-GPU-solver1.prototxt -gpu=0,1,2,3
