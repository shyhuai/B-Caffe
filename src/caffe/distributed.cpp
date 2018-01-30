#include "caffe/caffe.hpp"
#include "caffe/distributed.hpp"
#include "caffe/util/gpu_memory.hpp"

//#ifdef USE_MPI
#include <mpi.h> 
//#endif

namespace caffe {

DistManager::DistManager(shared_ptr<Solver> root_solver, const vector<int>& gpus, int nranks) :
    nranks_(nranks),
    root_solver_(root_solver),
    rank_(-1),
    reduce_counter_(1)
{
    gpus_ = gpus;
    Init();
}

DistManager::~DistManager() 
{
    CUDA_CHECK(cudaFreeHost(learnable_params_cpu_)); 
    CUDA_CHECK(cudaFreeHost(learnable_params_cpu_out_)); 
    delete semaphore_;
    delete p2pmanager_;
}

int DistManager::rank() {
    if (rank_ == -1) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        rank_ = world_rank;
    }
    return rank_;
}

void DistManager::Init()
{
    for (auto p = gpus_.begin(); p != gpus_.end(); ++p) {
        BlockingQueue<std::pair<int, size_t>> *nq = new BlockingQueue<std::pair<int, size_t>>();
        notify_queues_.push_back(nq);
    }
    semaphore_ = new Semaphore(gpus_.size());
    p2pmanager_ = new caffe::P2PManager(root_solver_, gpus_.size(), root_solver_->param(), this);
}

void DistManager::Run()
{
    p2pmanager_->Run(gpus_);
}

void DistManager::SolverInitialized(shared_ptr<Solver> solver, int inter_rank)
{
    LOG(INFO) << "DistManager receive initialized message, inter_rank: " << inter_rank;
    solver->net()->set_dist_queue(notify_queues_[inter_rank]);
    if (solver == root_solver_) {
        LOG(INFO) << "At root solver, inter rank: " << rank();
        size_t learnable_space_size = root_solver_->net()->learnable_space_size(0);
        CUDA_CHECK(cudaMallocHost(&learnable_params_cpu_, learnable_space_size));
        CUDA_CHECK(cudaMallocHost(&learnable_params_cpu_out_, learnable_space_size));
        //learnable_params_cpu_ = (void *)malloc(learnable_space_size);
        //learnable_params_cpu_out_ = (void *)malloc(learnable_space_size);
        if (!learnable_params_cpu_) {
            LOG(ERROR) << "Malloc cpu buffer error! Abort";
            exit(1);
        }
        reduce_thread0_.reset(new boost::thread(&DistManager::ReduceLoop, this, 0));
    } else {
        LOG(INFO) << "At normal solver";
    }
    solvers_.push_back(solver);
}

void DistManager::GetReduceBucketId(int type_id, int &id_from, size_t &count)
{
    LOG(INFO) << "Fetching reduce bucket id";
    id_from = Net::HOLD_ON_REDUCE; count = 0UL;
    int param_id = reduction_queue_[type_id][0].pop(); // wait here
    if (param_id == Net::END_OF_TRAIN || param_id == Net::END_OF_ITERATION) {
    } else {
        au_ids_.push_back(param_id);
    }
    size_t cnt = 0UL;
    for (auto p = au_ids_.begin(); p != au_ids_.end(); ++p) {
        int param_id = *p;
        cnt += root_solver_->net()->lp_aligned_count(param_id);
        id_from = param_id;
    }
    if (cnt > 0 || param_id == Net::END_OF_TRAIN || param_id == Net::END_OF_ITERATION) {
        count = cnt;
        id_from = param_id;
    } else {
        count = 0;
        id_from = Net::HOLD_ON_REDUCE;
    }
}

void DistManager::ParamIdPushed(int type_id, const int param_id, int inner_rank) 
{
    if (inner_rank == 0) {
        reduction_queue_[type_id][inner_rank].push(param_id);
        LOG(INFO) << "ParamIdPushed: " << " type_id: " << type_id << ", param_id: " << param_id << ", inner_rank: " << inner_rank;
    }
}

void DistManager::ReduceLoop(int type_id) 
{
    CUDA_CHECK(cudaSetDevice(gpus_[0]));

    LOG(INFO) << "Reduce loop started...";
    while(1) {
        LOG(INFO) << "At reduce looping...";
        int id_from;
        size_t count;
        GetReduceBucketId(type_id, id_from, count);
        if (id_from == Net::HOLD_ON_REDUCE)  {
            continue;
        }         
        LOG(INFO) << "Prepare to reduce..., solver size: " << solvers_.size();
        int real_id_from = id_from;
        if (id_from != Net::END_OF_TRAIN) {
            real_id_from = au_ids_[au_ids_.size()-1];
        }
        au_ids_.clear();
        for (int i = 0; i < solvers_.size(); ++i) {
            BlockingQueue<std::pair<int, size_t>>* nq = notify_queues_[i];
            nq->push(std::make_pair(real_id_from, count));
        }
        if (id_from == Net::END_OF_TRAIN) {
            break;
        }

        // 1. Start to reduce
        // 1.1 Wait all GPUs finished
        LOG(INFO) << "Waiting for all threads NCCL Allreduce...";
        semaphore_->WaitAll();

        LOG(INFO) << "Start to copy from GPU to CPU";
        // 2. Copy from GPU0 (root_solver) to CPU
        caffe_gpu_memcpy(count, root_solver_->net()->learnable_params_ptr(type_id)[real_id_from], learnable_params_cpu_);

        // 3. MPI_allreduce
        Allreduce(count);
        
        // 4. Copy back from CPU to GPU0,1,...
        LOG(INFO) << "Copy back from CPU to GPU0,1,...";
        for (auto id = solvers_.begin(); id != solvers_.end(); ++id) {
            const shared_ptr<Solver> solver = *id;
            //caffe_gpu_memcpy(count, learnable_params_cpu_out_, solver->net()->learnable_params_ptr(type_id)[id_from]);
            caffe_gpu_memcpy(count, learnable_params_cpu_out_, solver->net()->learnable_params()[real_id_from]->current_mutable_data_memory(true));
            solver->iteration_complete_signal(type_id);
        }
    }
}


void DistManager::Allreduce(int count)
{
    LOG(INFO) << "MPI Allreduce... counter: " << reduce_counter_;
    reduce_counter_++;
    MPI_Allreduce(
            learnable_params_cpu_,
            learnable_params_cpu_out_,
            count,
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD);
}

}  // namespace caffe
