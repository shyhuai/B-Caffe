#CUDA_VISIBLE_DEVICES=2,3 caffe train -solver=resnet-b128-GPU-solver.prototxt -gpu=0,1
#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 2 -hostfile cluster2 caffe train -solver=resnet-b128-GPU-solver.prototxt -gpu=0,1 >tmp2o.log 2>&1
#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 4 -hostfile cluster4 caffe train -solver=resnet-b128-GPU-solver.prototxt -gpu=0,1 >tmp4o.log 2>&1
#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 8 -hostfile cluster8 caffe train -solver=resnet-b128-GPU-solver.prototxt -gpu=0,1

#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 2 -hostfile cluster2 caffe train -solver=solver.prototxt -gpu=0,1 >tmp2o.log 2>&1
#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 4 -hostfile cluster4 caffe train -solver=solver.prototxt -gpu=0,1 >tmp4o.log 2>&1
#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 8 -hostfile cluster8 caffe train -solver=solver.prototxt -gpu=0,1 >tmp8o.log 2>&1
mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 4 -hostfile cluster4 caffe train -solver=solver.prototxt -gpu=0,1 >sync-convergence-4nodes.log 2>&1 #>tmp4s.log 2>&1
#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 2 -hostfile cluster2 caffe train -solver=solver.prototxt -gpu=0,1 >convergence-2nodes.log 2>&1 #>tmp4s.log 2>&1
