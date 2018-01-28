#ifndef DISTRIBUTED_HPP_
#define DISTRIBUTED_HPP_

#include "caffe/solver.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/thread_pool.hpp"


namespace caffe {

class Semaphore;

class DistManager {
 public:
  DistManager(shared_ptr<Solver> root_solver, const vector<int>& gpus, int nranks);
  ~DistManager();

  void Init();
  void Run();
  void Allreduce(int count);
  int rank();

  // Another thread
  void GetReduceBucketId(int type_id, int &id_from, size_t &count);
  void ReduceLoop(int type_id);

  // Callback
  void SolverInitialized(shared_ptr<Solver> solver, int inter_rank);


 protected:
  const size_t nranks_;
  shared_ptr<Solver> root_solver_;
  int rank_;

  vector<shared_ptr<Solver>> solvers_;
  caffe::P2PManager* p2pmanager_;
  vector<int> gpus_;
  vector<BlockingQueue<std::pair<int, size_t>>*> notify_queues_;
  std::vector<int> au_ids_;

  Semaphore *semaphore_;
  void *learnable_params_cpu_;

  unique_ptr<boost::thread> reduce_thread0_;
  unique_ptr<boost::thread> reduce_thread1_;
  BlockingQueue<int> reduction_queue_[2];

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

}// namespace caffe
#endif
