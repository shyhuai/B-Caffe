#include "caffe/caffe.hpp"
#include "caffe/distributed.hpp"
#include "caffe/util/gpu_memory.hpp"

//#ifdef USE_MPI
#include <mpi.h> 
//#endif
#include <algorithm>    // std::max

namespace caffe {

DistManager::DistManager(shared_ptr<Solver> root_solver, const vector<int>& gpus, int nranks) :
    nranks_(nranks),
    root_solver_(root_solver),
    rank_(-1),
    reduce_counter_(1),
    benchmark_(true),
    iter_(0)
{
    gpus_ = gpus;
    gradients_merged_ = false;
    Init();
}

DistManager::~DistManager() 
{
    CUDA_CHECK(cudaFreeHost(learnable_params_cpu_)); 
    CUDA_CHECK(cudaFreeHost(learnable_params_cpu_out_)); 
    delete semaphore_;
    delete p2pmanager_;
    delete merged_param_;
    for (int i = 0; i < overheads_.size(); i++) {
        delete overheads_[i];
    }
}

int DistManager::rank() {
    if (rank_ == -1) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        rank_ = world_rank;
    }
    return rank_;
}

int DistManager::GetWorldRank() {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    return world_rank;
}

int DistManager::GetWorldSize() {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  return world_size;
}

void DistManager::Init()
{
    for (auto p = gpus_.begin(); p != gpus_.end(); ++p) {
        BlockingQueue<std::pair<int, size_t>> *nq = new BlockingQueue<std::pair<int, size_t>>();
        notify_queues_.push_back(nq);
    }
    semaphore_ = new Semaphore(gpus_.size());
    p2pmanager_ = new caffe::P2PManager(root_solver_, gpus_.size(), root_solver_->param(), this);
    gradients_merged_ = root_solver_->param().gradients_merged();
    merged_param_ = new MergedParam();
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
        LOG(INFO) << "learnable_space_size: " << learnable_space_size;
        CUDA_CHECK(cudaMallocHost(&learnable_params_cpu_, learnable_space_size));
        CUDA_CHECK(cudaMallocHost(&learnable_params_cpu_out_, learnable_space_size));
        //learnable_params_cpu_ = (void *)malloc(learnable_space_size);
        //learnable_params_cpu_out_ = (void *)malloc(learnable_space_size);
        if (!learnable_params_cpu_) {
            LOG(ERROR) << "Malloc cpu buffer error! Abort";
            exit(1);
        }

        const vector<int>& learnable_param_ids = root_solver_->net()->learnable_param_ids();

        //LOG(INFO) << "----------------------- param_ids: ";
        for (int i = 0; i < learnable_param_ids.size(); i++) {
            int param_id = learnable_param_ids[i];
            size_t count = root_solver_->net()->lp_aligned_count(param_id);
            Overhead *overhead = new Overhead(param_id, count);
            overheads_.push_back(overhead);
            param_id_indexes_[param_id] = i;
            //LOG(INFO) << "["<<i<<"]:"<<learnable_param_ids[i];
        }
        //print_overheads();
        //LOG(INFO) << "----------------------- param_ids end";
        reduce_thread0_.reset(new boost::thread(&DistManager::ReduceLoop, this, 0));
    } else {
        LOG(INFO) << "At normal solver";
    }
    solvers_.push_back(solver);
}

void DistManager::print_overheads() {
    if (rank() == 0) {
        LOG(INFO) << "------ overheads: [id, size, compute_time, communication_time] at rank: " << rank();
        for (int i = 0; i < overheads_.size(); i++) {
            Overhead *overhead = overheads_[i];
            LOG(INFO) << "[" << overhead->param_id_ << ", " << overhead->size_ << ", "<< overhead->compute_time_ << ", " << overhead->communication_time_ << ", " << overhead->merged_time_ << "]";
        }
    }
}

void DistManager::merge_overheads() {
    //if (rank() == 0) {
        LOG(INFO) << "in merge_overheads.";
        Overhead *overhead0 = overheads_[0];
        start_param_map_[overhead0->param_id_] = overhead0->param_id_;
        for (int i = 1; i < overheads_.size(); i++) {
            Overhead *overhead1 = overheads_[i];
            if (overhead1->compute_time_ == overhead0->compute_time_) {
                overhead0->communication_time_ += overhead1->communication_time_;
                overhead0->size_ += overhead1->size_;
                overhead1->size_ = 0;
                start_param_map_[overhead1->param_id_] = overhead0->param_id_;
            } else {
                overhead0 = overhead1;
                start_param_map_[overhead0->param_id_] = overhead0->param_id_;
            }
        }
        vector<Overhead *>::iterator it = overheads_.begin();
        for ( ; it != overheads_.end(); ) {
            Overhead *overhead = *it;
            if (overhead->size_ == 0) {
                delete *it;
                it = overheads_.erase(it);
            } else {
                ++it;
            }
        }
        std::reverse(overheads_.begin(), overheads_.end());
        print_overheads();
    //}
}


double DistManager::predict_comm_time(size_t size, int n)
{
    if (n == 8) {
        double x1 = 16080;
        double y1 = 932;
        double x2 = 521092;
        double y2 = 2182;

        double x3 = 523474;
        double y3 = 3125;
        double x4 = 1026100;
        double y4 = 3992;
        double p = 0.0;
        if (size < x2) {
            p = (y1-y2)/(x1-x2) * (size-x2) + y2; 
        } else {
            p = (y3-y4)/(x3-x4) * (size-x4) + y4;
        }
        return p+5000.0;
    }
    return 0;
}


int DistManager::search_index(int startIdx, vector<Overhead *>& overheads)
{
    double sumComm = 0.0;
    double sumComp = 0.0;
    for (int i = startIdx; i < overheads.size() - 1; i++) {
        Overhead* overhead = overheads[i];
        Overhead* overhead2 = overheads[i+1];
        double comm = overhead->communication_time_;
        double comp = overhead2->compute_time_;
        sumComm += comm;
        sumComp += comp;
        if (sumComp >= sumComm) {
            return i; // Case 1 and Case 2: optimal.
        }
    }
    return -1; // Case 3, which needs to merge gradients.
}

void DistManager::comm_start(const vector<double> &tc, const vector<double> &tb, const vector<double> &taub, int L, vector<double> &tauc)
{
    tauc[L-1] = taub[L-1]+tb[L-1];
    for (int l = L-2; l >= 0; l--) {
        tauc[l] = std::max(tauc[l+1]+tc[l+1], taub[l]+tb[l]);
    }
}
#define alpha_p 20.
//#define alpha_p 6.25 
double DistManager::allreduce_time(size_t size)
{
    double alpha = alpha_p*nranks_;
    double beta = 8/(10.*1e9);
    double gamma = 0.;
    int n = nranks_;
    size = size * 4;
    double comm = 2*n*alpha + 2*(n-1)*size*beta/n + (n-1)*size*gamma/n;
    return comm;
}

void DistManager::generate_merged_param()
{
    merge_overheads();
    vector<int> param_ids;
    vector<double> tb;
    vector<double> tc;
    vector<size_t> p; 
    ///2*n*alpha + 2*(n-1)*M*beta/n + (n-1)*M*gamma/n
    double alpha = alpha_p*nranks_;
    for (int i = 0; i < overheads_.size(); i++) {
        Overhead *o = overheads_[i];
        if (o->compute_time_ > 0) {
            tb.push_back(o->compute_time_);
            p.push_back(o->size_);
            double comm = allreduce_time(o->size_);
            tc.push_back(comm);
            param_ids.push_back(o->param_id_);
        }
    }
    int L = tb.size(); 
    vector<double> taub(L);
    vector<double> tauc(L);
    double tf = 0.0;
    taub[L-1] = tf;
    for (int l = L-2; l>=0; l--) {
        taub[l] = taub[l+1] + tb[l+1];
    }
    // calculate tauc
    comm_start(tc, tb, taub, L, tauc);
    for (int l = L-1; l >= 1; l--) {
        double tmp = 0.0;
        if (l > 1) {
            tmp = taub[l-2];
        } else {
            tmp = taub[0]+tb[0];
        }
        if (tmp - tauc[l] < 2*nranks_ * alpha) {
            // MERGE here
            tc[l] = 0.0;
            p[l-1] = p[l-1]+p[l];
            tc[l-1] = allreduce_time(p[l-1]);
            comm_start(tc, tb, taub, L, tauc);
            M_.insert(param_ids[l]);
        }
    }
    if (rank() == 0) {
        LOG(INFO) << "=============== " << rank() << " ============";
        merged_param_->print();

    }
    count_ = M_.size();
    MPI_Bcast(&count_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *M = new int[count_];
    int i = 0;
    if (rank() == 0) {
        for (auto m:M_) {
            M[i++] = m;
        }
    }
    // Bcast to other mpi processes.
    MPI_Bcast(M, count_, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank() != 0) {
        for (i = 0; i < count_; i++) {
            M_.insert(M[i]);
        }
    }
    //delete M;
    LOG(INFO) << "=============== M at rank(): " << rank() << "============";
    for(auto m : M_) {
        LOG(INFO) << m;
    } 

    //std::reverse(overheads_.begin(), overheads_.end());
    print_overheads();
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
    //if (au_ids_.size() % 2 != 0) {
    //    id_from = Net::HOLD_ON_REDUCE;
    //    return;
    //}
    size_t cnt = 0UL;
    for (auto p = au_ids_.begin(); p != au_ids_.end(); ++p) {
        int tmp_param_id = *p;
        cnt += root_solver_->net()->lp_aligned_count(tmp_param_id);
        id_from = tmp_param_id;
        //if (tmp_param_id % 2 == 0) {
        //    id_from = tmp_param_id;
        //}
        //LOG(INFO) << "ParamId in queue: " << tmp_param_id;
    }
    if (param_id == Net::END_OF_TRAIN || param_id == Net::END_OF_ITERATION) {
        count = cnt;
        id_from = param_id;
        return;
    }
    if (!gradients_merged_) {
        count = cnt;
        id_from = param_id;
    } else {

        //if ((benchmark_ && cnt > 0) || cnt > 8192*16) { // GoogleNet
        //if ((benchmark_ && cnt > 0) || cnt > 8192*2) { // ResNet ??
        //if ((benchmark_ && cnt > 0) || cnt > 1e5) { // VGG
        //if ((benchmark_ && cnt > 0) || (!benchmark_ && cnt > 8192*32 && param_id > 25) || (!benchmark_&&param_id == 0 && cnt>0)) { // VGG special case
        if (benchmark_) {
            if (cnt > 0) {
                count = cnt;
                id_from = param_id;
            } else {
                count = 0;
                id_from = Net::HOLD_ON_REDUCE;
            }
        } else {
            int tmp_param_id = start_param_map_[param_id];
            if (M_.count(tmp_param_id) == 0 && cnt > 0) {
            //if (param_id== 0 && cnt > 0) { // SyncEASGD
            //if (cnt > 8192*16) {
                count = cnt;
                id_from = param_id;
            } else {
                count = 0;
                id_from = Net::HOLD_ON_REDUCE;
            }
        }
    }
}

void DistManager::ParamIdPushed(int type_id, const int param_id, int inner_rank, double time) 
{
    if (inner_rank == 0) {
        reduction_queue_[type_id][inner_rank].push(param_id);
        if (benchmark_ && param_id >= 0) {
            Overhead *overhead = overheads_[param_id_indexes_[param_id]];
            overhead->set_compute_time(time);
        }
        //LOG(INFO) << "ParamIdPushed: " << " type_id: " << type_id << ", param_id: " << param_id << ", inner_rank: " << inner_rank;
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
        LOG(INFO) << "Fetched id_from: " << id_from << ", count: " << count; 
        if (id_from == Net::HOLD_ON_REDUCE)  {
            continue;
        }         
        LOG(INFO) << "Prepare to reduce..., solver size: " << solvers_.size() << ", au_ids size: " << au_ids_.size();
        int real_id_from = id_from;
        if (id_from == Net::END_OF_TRAIN) {
            for (int i = 0; i < solvers_.size(); ++i) {
                BlockingQueue<std::pair<int, size_t>>* nq = notify_queues_[i];
                nq->push(std::make_pair(Net::END_OF_TRAIN, 0));
            }
            break;
        }

        if (id_from != Net::END_OF_TRAIN && au_ids_.size() > 0) {
            real_id_from = au_ids_.back();
        }
        if (real_id_from >= 0) {
            LOG(INFO) << "real_id_from: " << real_id_from; 
            for (int i = 0; i < solvers_.size(); ++i) {
                BlockingQueue<std::pair<int, size_t>>* nq = notify_queues_[i];
                nq->push(std::make_pair(real_id_from, count));
            }
            // 1. Start to reduce
            // 1.1 Wait all GPUs finished
            LOG(INFO) << "Waiting for all threads NCCL Allreduce...";
            semaphore_->WaitAll();

            LOG(INFO) << "Start to copy from GPU to CPU";
            // 2. Copy from GPU0 (root_solver) to CPU
            allreduce_timer_.Start();
            CUDA_CHECK(cudaSetDevice(gpus_[0]));
            size_t data_size = count * root_solver_->net()->lp_size(real_id_from);
            caffe_gpu_memcpy(data_size, root_solver_->net()->learnable_params_ptr(type_id)[real_id_from], learnable_params_cpu_);

            // 3. MPI_allreduce
            Allreduce(count);

            // 4. Update model
            // learnable_params_[id_from]->diff_type()
            LOG(INFO) << "Update model in the cpu side";
            Tensor::cpu_scal(count, root_solver_->net()->learnable_params()[real_id_from]->diff_type(), learnable_params_cpu_out_, 1.F / (Caffe::solver_count() * root_solver_->net()->global_grad_scale() * nranks_));

            // 5. Copy back from CPU to GPU0,1,...
            LOG(INFO) << "Copy back from CPU to GPU0,1,... real_id_from: " << real_id_from;
            for (int i = 0; i < solvers_.size(); ++i) {
                const shared_ptr<Solver> solver = solvers_[i];
                int gpu_id = gpus_[i];
                CUDA_CHECK(cudaSetDevice(gpu_id));
                //caffe_gpu_memcpy(count, learnable_params_cpu_out_, solver->net()->learnable_params()[real_id_from]->current_mutable_data_memory(true));
                caffe_gpu_memcpy(data_size, learnable_params_cpu_out_, solver->net()->learnable_params_ptr(type_id)[real_id_from]);
            }

            for (int i = 0; i < solvers_.size(); ++i) {
                BlockingQueue<std::pair<int, size_t>>* nq = notify_queues_[i];
                for (int id_reduced: au_ids_) {
                    nq->push(std::make_pair(Net::END_OF_REDUCE, id_reduced));
                }
            }
        }

        if (id_from == Net::END_OF_ITERATION) {
            iter_++;
            if (benchmark_ && iter_ > 3) {
                //print_overheads();
                if (gradients_merged_) {
                    generate_merged_param();
                }
                benchmark_ = false;
            }

            for (int i = 0; i < solvers_.size(); ++i) {
                BlockingQueue<std::pair<int, size_t>>* nq = notify_queues_[i];
                nq->push(std::make_pair(Net::END_OF_ITERATION, 0));
            }
        }

        double time = allreduce_timer_.MicroSeconds();
        if (benchmark_) {
            Overhead *overhead = overheads_[param_id_indexes_[real_id_from]];
            overhead->set_communication_time(time);
        } else {
            Overhead *overhead = overheads_[param_id_indexes_[real_id_from]];
            overhead->set_merged_time(time);
        }
        au_ids_.clear();
    }
    print_overheads();
}


void DistManager::Allreduce(int count)
{
    LOG(INFO) << "MPI Allreduce... counter: " << reduce_counter_;
    reduce_counter_++;
#if DEBUG
    float *tmp = (float *)learnable_params_cpu_;
    DLOG(INFO) << "MPIAllReduce before: " << tmp[0] << ", " <<tmp[1];
#endif
    MPI_Allreduce(
            learnable_params_cpu_,
            learnable_params_cpu_out_,
            count,
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD);
#if DEBUG
    tmp = (float *)learnable_params_cpu_out_;
    DLOG(INFO) << "MPIAllReduce after: " << tmp[0] << ", " <<tmp[1];
#endif
}

}  // namespace caffe
