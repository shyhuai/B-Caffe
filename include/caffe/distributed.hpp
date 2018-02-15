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

}// namespace caffe
#endif
