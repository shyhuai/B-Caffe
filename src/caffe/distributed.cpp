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

void DistManager::generate_merged_param()
{
    merge_overheads();
    for (int i = 0; i < overheads_.size()-1;) {
        int merged_idx = search_index(i, overheads_);
        if (merged_idx >= 0) {
            LOG(INFO) << "Found Case 1 or Case 2! Merged_idx: " << merged_idx;
            for (int j = i; j <= merged_idx; j++) {
                merged_param_->push_param_id(overheads_[j]->param_id_, -1, overheads_[j]->size_);
            }
            i = merged_idx + 1;
        } else {
            // Need to merge here
            double merged_comm = overheads_[i]->communication_time_;
            size_t sum_size = overheads_[i]->size_;
            double sum_comp = overheads_[i+1]->compute_time_;
            bool hidden_found = false;
            int j;
            for (j = i+1; j < overheads_.size() - 1; j++) {
                Overhead* overhead = overheads_[j];
                Overhead* overhead2 = overheads_[j+1];
                double comp = overhead2->compute_time_;
                size_t size = overhead->size_;
                sum_size += size;
                double predict_comm = predict_comm_time(sum_size, nranks_);
                LOG(INFO) << "predict_comm: " << predict_comm << ", comm: " << overhead->communication_time_;
                merged_comm = predict_comm + sum_comp;
                sum_comp += comp;
                if (merged_comm <= sum_comp) {
                    // Can be hidded after merged
                    hidden_found = true;
                    break;
                }
            }
            if (hidden_found) {
                // Case 3
                LOG(INFO) << "Found Case 3! Param_id: " << overheads_[i]->param_id_;
                merged_param_->push_param_id(overheads_[i]->param_id_, -1, overheads_[i]->size_);
                for (int k = i+1; k <= j; k++) {
                    merged_param_->push_param_id(overheads_[k]->param_id_, 1, overheads_[k]->size_);
                }
            } else {
                // Case 4
                LOG(INFO) << "Found Case 4! Param_id: " << overheads_[i]->param_id_;
                sum_size = overheads_[i]->size_;
                merged_param_->push_param_id(overheads_[i]->param_id_, -1, overheads_[i]->size_);
                double sum_comm = overheads_[i]->communication_time_;
                sum_comp = overheads_[i+1]->compute_time_;
                for (j = i+1; j < overheads_.size()-1; j++) {
                    Overhead* overhead = overheads_[j];
                    Overhead* overhead2 = overheads_[j+1];
                    double comm = overhead->communication_time_;
                    double comp = overhead2->compute_time_;
                    size_t size = overhead->size_;
                    sum_size += size;
                    sum_comm += comm;
                    double predict_comm = predict_comm_time(sum_size, nranks_);
                    LOG(INFO) << "predict_comm: " << predict_comm << ", sum_comp: " << sum_comp << ", sum_comm: " << sum_comm; 
                    if (sum_comp + predict_comm > sum_comm) {
                        j--;
                        break;
                    } else {
                        merged_param_->push_param_id(overheads_[j]->param_id_, 1, overheads_[j]->size_);
                    }
                    sum_comp += comp;
                }
            }
            i = j+1;
        }
    }
    //if (rank() == 0) {
        merged_param_->print();
    //}
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
        if ((benchmark_ && cnt > 0) || cnt > 8192*2) { // ResNet ??
            //if ((benchmark_ && cnt > 0) || (param_id % 2 != 0 && cnt > 0)) {
            count = cnt;
            id_from = param_id;
        } else {
            count = 0;
            id_from = Net::HOLD_ON_REDUCE;
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
            caffe_gpu_memcpy(count, root_solver_->net()->learnable_params_ptr(type_id)[real_id_from], learnable_params_cpu_);

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
                caffe_gpu_memcpy(count, learnable_params_cpu_out_, solver->net()->learnable_params_ptr(type_id)[real_id_from]);
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
                print_overheads();
                benchmark_ = false;
                //generate_merged_param();
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
    //print_overheads();
    generate_merged_param();
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
