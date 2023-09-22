// Run executable with `OMP_NUM_THREADS=4 OMP_PLACES=threads OMP_PROC_BIND=true OMP_DISPLAY_AFFINITY=true likwid_test`
// Output should look similar:
//
//Results for thread 0 (hw thread 1):
//Event ACTUAL_CPU_CLOCK:                 422688
//Event MAX_CPU_CLOCK:            632121
//Event RETIRED_INSTRUCTIONS:             179777
//Event CPU_CLOCKS_UNHALTED:              125078
//Event RETIRED_SSE_AVX_FLOPS_SINGLE_ALL:                 1024
//Event MERGE:            0
//Results for thread 1 (hw thread 3):
//Event ACTUAL_CPU_CLOCK:                 614119
//Event MAX_CPU_CLOCK:            658182
//Event RETIRED_INSTRUCTIONS:             336703
//Event CPU_CLOCKS_UNHALTED:              220149
//Event RETIRED_SSE_AVX_FLOPS_SINGLE_ALL:                 1024
//Event MERGE:            0
//Results for thread 2 (hw thread 5):
//Event ACTUAL_CPU_CLOCK:                 517851
//Event MAX_CPU_CLOCK:            581742
//Event RETIRED_INSTRUCTIONS:             326627
//Event CPU_CLOCKS_UNHALTED:              200616
//Event RETIRED_SSE_AVX_FLOPS_SINGLE_ALL:                 1024
//Event MERGE:            0
//Results for thread 3 (hw thread 7):
//Event ACTUAL_CPU_CLOCK:                 554137
//Event MAX_CPU_CLOCK:            629097
//Event RETIRED_INSTRUCTIONS:             126078
//Event CPU_CLOCKS_UNHALTED:              189136
//Event RETIRED_SSE_AVX_FLOPS_SINGLE_ALL:                 1024


#include "sycl/sycl.hpp"
#include <likwid-marker.h>
#include <string>
#include <likwid.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <vector>

class LikwidPerfctrResult {
public:
    std::vector<std::vector<double>> results;
    std::vector<std::vector<double>> tmp;

    const int num_threads, num_events;

    LikwidPerfctrResult(int nthreads, int nevents) : results(nthreads, std::vector<double>(nevents)),
                                                     tmp(nthreads, std::vector<double>(nevents)),
                                                     num_threads(nthreads), num_events(nevents) {
    }
};

class LikwidPerftool : public hipsycl::ext::performance_tool_api {
    int groupId = -1;
    char const* event_group;
    int *hwthreads;
    LikwidPerfctrResult *results;
public:
    LikwidPerftool(char const* event_group) : event_group(event_group) {
        int nthreads = omp_get_max_threads();
        std::cout << nthreads << std::endl;
        hwthreads = new int[nthreads];
    }

    ~LikwidPerftool() {
        delete results;
        delete hwthreads;
        perfmon_finalize();
    }

    virtual void init(hipsycl::rt::device_id device_id) override {
        int nthreads = omp_get_max_threads();
#pragma omp parallel shared(hwthreads)
        {
            if (getcpu((unsigned int*) (hwthreads + omp_get_thread_num()), NULL) != 0)
            {
                throw std::runtime_error("Cannot get cpu id on thread " + std::to_string(omp_get_thread_num()) + ", errno: " + std::to_string(errno));
            }
#pragma omp critical
            std::cout << "Thread " << omp_get_thread_num() << ": hw thread " << hwthreads[omp_get_thread_num()] << ", tid: " << gettid() << "\n";
        }

        perfmon_init(nthreads, (int *) hwthreads);
        groupId = perfmon_addEventSet(event_group);
        if (perfmon_setupCounters(groupId) < 0)
        {
            throw std::runtime_error("Cannot setup hardware counters!");
        }
        int nevents = perfmon_getNumberOfEvents(groupId);
        results = new LikwidPerfctrResult(nthreads, nevents);
        perfmon_startCounters();
    }

    virtual void kernel_start(std::type_info const& kernel_type_info) override {
    }
    virtual void kernel_end(std::type_info const& kernel_type_info) override {
    }

    virtual void omp_thread_start(std::type_info const& kernel_type_info) override {
        int thread_num = omp_get_thread_num();
#pragma omp critical
        { std::cout << "start thread " << thread_num << ", hw thread: " << sched_getcpu() << ", tid: " << gettid() << "\n"; }
        perfmon_readCountersCpu(hwthreads[thread_num]);
        for (int i = 0; i < results->num_events; i++) {
            results->tmp[thread_num][i] = perfmon_getLastResult(groupId, i, thread_num);
        }
    }
    virtual void omp_thread_end(std::type_info const& kernel_type_info) override {
        int thread_num = omp_get_thread_num();
#pragma omp critical
        { std::cout << "stop thread " << thread_num << ", hw thread: " << sched_getcpu() << ", tid: " << gettid() << "\n"; }
        perfmon_readCountersCpu(hwthreads[thread_num]);
        for (int i = 0; i < results->num_events; i++) {
            results->results[thread_num][i] += perfmon_getLastResult(groupId, i, thread_num);
        }
    }

    void print_results() {
        for (int i = 0; i < results->num_threads; i++) {
            std::cout << "Results for thread " << i << " (hw thread " << hwthreads[i] << "):\n";
            for (int j = 0; j < results->num_events; j++) {
                char *event_name = perfmon_getEventName(groupId, j);
                std::cout << "Event " << event_name << ": \t\t" << results->results[i][j] << "\n";
            }
        }
    }
};

class Triad;

int main()
{
    constexpr size_t len = 1024;

    // LIKWID_MARKER_INIT;

    std::shared_ptr<LikwidPerftool> likwid = std::make_shared<LikwidPerftool>("FLOPS_SP");
    sycl::queue q(sycl::property_list{sycl::property::queue::hipSYCL_instrumentation(likwid)});

    sycl::buffer<float> a(len);
    sycl::buffer<float> b(len);
    sycl::buffer<float> c(len);
    sycl::buffer<float> result(len);

    {
        auto&& aAcc = a.get_host_access();
        auto&& bAcc = b.get_host_access();
        auto&& cAcc = c.get_host_access();

        for (size_t i = 0; i < len; i++) {
            aAcc[i] = 1.0;
            bAcc[i] = 2.0;
            cAcc[i] = -3.0;
        }
    }

    q.submit([&](sycl::handler& cg) {
        auto&& aAcc = a.get_access<sycl::access_mode::read>(cg);
        auto&& bAcc = b.get_access<sycl::access_mode::read>(cg);
        auto&& cAcc = c.get_access<sycl::access_mode::read>(cg);
        auto&& resultAcc = result.get_access<sycl::access_mode::write>(cg);

        cg.parallel_for<Triad>(sycl::range<1>{len}, [=](sycl::id<1> i) {
            resultAcc[i] = aAcc[i] + bAcc[i] * cAcc[i];
        });
    });

    q.wait();
    q.submit([&](sycl::handler& cg) {
        auto&& aAcc = a.get_access<sycl::access_mode::read>(cg);
        auto&& bAcc = b.get_access<sycl::access_mode::read>(cg);
        auto&& cAcc = c.get_access<sycl::access_mode::read>(cg);
        auto&& resultAcc = result.get_access<sycl::access_mode::write>(cg);

        cg.parallel_for<Triad>(sycl::range<1>{len}, [=](sycl::id<1> i) {
            resultAcc[i] = aAcc[i] + bAcc[i] * cAcc[i];
        });
    });

    q.wait();

    likwid->print_results();
    // LIKWID_MARKER_CLOSE;

    return 0;
}