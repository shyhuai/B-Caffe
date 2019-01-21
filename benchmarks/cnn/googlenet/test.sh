#mpirun --mca oob_tcp_static_ipv4_ports 5924-5928 --mca btl_tcp_if_include bond0 --np 2 -hostfile cluster2 /home/comp/csshshi/repositories/nvcaffe-dist/build/tools/caffe #train -solver=solver.prototxt -gpu=0,1 #>tmp2s.log 2>&1 
#mpirun --np 2 --hostfile cluster2 caffe train -solver=solver.prototxt -gpu=0,1 >convergence-2nodes.log 2>&1 
#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 4 -hostfile cluster4 caffe train -solver=solver.prototxt -gpu=0,1 >tmp4s.log 2>&1
#MPIPATH=/home/comp/csshshi/local/openmpi3.1.1
MPIPATH=/home/comp/csshshi/local/pgiopenmpi
nnodes=2
$MPIPATH/bin/mpirun --prefix $MPIPATH -x CUDA_VISIBLE_DEVICES=0,1 -np $nnodes -hostfile cluster$nnodes caffe train -solver=solver.prototxt -gpu=0,1 #>convergence-4nodes.log 2>&1 #>tmp4s.log 2>&1
#mpirun -x CUDA_VISIBLE_DEVICES=2,3 -np 8 -hostfile cluster8 caffe train -solver=solver.prototxt -gpu=0,1 >tmp8s.log 2>&1
