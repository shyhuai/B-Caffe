#ifndef DISTRIBUTED_HPP_
#define DISTRIBUTED_HPP_

#include "caffe/solver.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/thread_pool.hpp"


namespace caffe {

class Semaphore;
class P2PManager;
class Overhead;
class MergedParam;

class DistManager {
 public:
  DistManager(shared_ptr<Solver> root_solver, const vector<int>& gpus, int nranks);
  ~DistManager();

  void Init();
  void Run();
  void Allreduce(int count);
  int rank();
  static int GetWorldRank();
  static int GetWorldSize();

  // Another thread
  void GetReduceBucketId(int type_id, int &id_from, size_t &count);
  void ReduceLoop(int type_id);

  // Callback
  void SolverInitialized(shared_ptr<Solver> solver, int inter_rank);
  void ParamIdPushed(int type_id, const int param_id, int inner_rank, double time);
  Semaphore* semaphore() {
      return semaphore_;
  }
  int nranks() {
      return nranks_;
  }
  void print_overheads();
  void merge_overheads();
  double predict_comm_time(size_t size, int n);
  int search_index(int startIdx, vector<Overhead *>& overheads);
  void generate_merged_param();

 protected:
  vector<Overhead *> overheads_;
  map<int, int> param_id_indexes_;
  const size_t nranks_;
  shared_ptr<Solver> root_solver_;
  int rank_;
  int reduce_counter_;
  bool benchmark_;
  int iter_;
  Timer allreduce_timer_;
  bool gradients_merged_;

  vector<shared_ptr<Solver>> solvers_;
  caffe::P2PManager* p2pmanager_;
  vector<int> gpus_;
  vector<BlockingQueue<std::pair<int, size_t>>*> notify_queues_;
  std::vector<int> au_ids_;

  Semaphore *semaphore_;
  void *learnable_params_cpu_;
  void *learnable_params_cpu_out_;

  unique_ptr<boost::thread> reduce_thread0_;
  unique_ptr<boost::thread> reduce_thread1_;
  BlockingQueue<int> reduction_queue_[2][4];
  MergedParam* merged_param_;
  map<int, int> start_param_map_; // [orignal_param_id] = layer_start_param_id;

  // For merged-gradient layers: M
  std::set<int> M_;
  int count_;
  void comm_start(const vector<double> &tc, const vector<double> &tb, const vector<double> &taub, int L, vector<double> &tauc);
  double allreduce_time(size_t size);

}; // class DistManager


class Semaphore {
  public:
   Semaphore(long count = 0)
       : count_(count), size_(count) {
       }

   void Signal() {
       boost::unique_lock<boost::mutex> lock(mutex_);
       ++count_;
       cv_.notify_one();
   }

   void Wait() {
       boost::unique_lock<boost::mutex> lock(mutex_);
       cv_.wait(lock, [=] { return count_ > 0; });
       --count_;
   }

   void WaitAll() {
       boost::unique_lock<boost::mutex> lock(mutex_);
       cv_.wait(lock, [=] { return count_ == size_; });
   }

  private:
   boost::mutex mutex_;
   boost::condition_variable cv_;
   long count_;
   long size_;
};

class Overhead {
    public:
        Overhead(int param_id, size_t size):
            param_id_(param_id), size_(size) {
                compute_time_ = 0.0;
                communication_time_ = 0.0;
                merged_time_ = 0.0;
            }
        ~Overhead() {}
        int param_id_;
        size_t size_;
        double compute_time_;
        double communication_time_;
        double merged_time_;
        void set_compute_time(double compute_time) {
            compute_time_ = compute_time;
        }
        void set_communication_time(double communication_time) {
            communication_time_ = communication_time;
        }
        void set_merged_time(double merged_time) {
            merged_time_ =  merged_time;
        }
};

class MergedParam {
    public:
        MergedParam():num_group_(0) {}
        ~MergedParam() {}
        int num_group_;
        vector<vector<int> *> merged_groups_;
        vector<size_t> group_sizes_;
        map<int, int> end_param_to_group_idx_; // [end_param_id] = group_idx;
        void push_param_id(int param_id, int merged_param_id, size_t size) {
            vector<int> *group;
            if (merged_param_id == -1) {
                group = new vector<int>();
                merged_groups_.push_back(group);
                group_sizes_.push_back(0);
                num_group_++;
            } else {
                group = merged_groups_[num_group_-1];
            }
            group->push_back(param_id);
            group_sizes_[num_group_-1] += size;
            end_param_to_group_idx_[param_id] = num_group_ - 1;
        }
        bool get_comm_param_id_and_size(int param_id, int& result_param_id, size_t& size) {
            bool is_exist = end_param_to_group_idx_.count(param_id);
            if (!is_exist) {
                return false;
            }
            int group_idx = end_param_to_group_idx_[param_id];
            result_param_id = merged_groups_[group_idx]->front();
            size = group_sizes_[group_idx];
            return true;
        }
        void print() {
            LOG(INFO) << "===============Printing Merged Param============";
            LOG(INFO) << "Number of groups: " << num_group_;
            for (int i = 0; i < merged_groups_.size(); i++) {
                vector<int>* group = merged_groups_[i];
                std::ostringstream stringStream;
                stringStream << "[" << i << "]: ";
                for (int j = 0; j < group->size(); j++) {
                    int param_id = group->at(j);
                    stringStream << param_id << ", ";
                }
                stringStream << "\n";
                LOG(INFO) << stringStream.str();
            }
            LOG(INFO) << "===============Merged Param End============";
        }
};

}// namespace caffe
#endif
